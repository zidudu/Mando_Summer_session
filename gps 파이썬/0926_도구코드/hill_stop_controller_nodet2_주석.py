#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math, csv, os
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int32, Bool, String
from tf.transformations import euler_from_quaternion

# ---------- 기본 파라미터 모음 (디폴트 값) ----------
DEFAULTS = dict(
    # ---------- IMU ----------
    tilt_threshold_deg=8.0,                # 피치 임계값(도) — 이 값 이상이면 '경사 게이트'가 참
    lpf_alpha=0.15,                        # LPF(지수평활) 계수(0..1). 작을수록 더 부드럽고 느림
    imu_topic='/zed/zed_nodelet/imu/data', # IMU 토픽 경로

    # ---------- 정지/재개 ----------
    stop_seconds=3.0,                      # 정지 유지 시간(초)
    resume_speed=5.0,                      # 재출발(버스트) 속도 (퍼블리시 값)
    resume_burst_seconds=0.6,              # 재출발 버스트 지속 시간(초)
    speed_topic='/hill_stop/speed_cmd',    # 이 노드가 퍼블리시할 속도 토픽

    # ---------- 웨이포인트 게이트 ----------
    wp_index_topic='/current_waypoint_index1',  # 위포 인덱스 수신 토픽
    target_waypoint_index=7,                    # 정지를 트리거할 목표 웨이포인트 인덱스
    require_exact_index=False,                  # True면 정확히 그 인덱스일 때만 정지, 
                                                # False이면현재 웨이포인트 인덱스가 target_wp_index 이상이면 모두 게이트를 True로 인식

    # ---------- 운용 옵션 ----------
    idle_publish=True,            # 평상시에도 keepalive 속도 퍼블리시할지 여부
    idle_default_speed=5.0,       # keepalive 퍼블리시 시 사용할 속도값
    debug_log=True,               # 디버깅 로그 출력 여부
    force_test_stop=False,        # 시작 직후 강제 정지(테스트용)

    # ---------- CSV ----------
    csv_path='raw_track_latlon_1.csv',  # speed 로그를 남길 CSV 경로

    # ---------- 신규: GRADEUP_ON ----------
    grade_enabled=True,                 # GRADEUP_ON 기능 사용 여부
    grade_topic='/gps/GRADEUP_ON',      # GRADEUP_ON 토픽 경로
    grade_pre_trigger_offset=2,         # 타깃 N이면 N-pre 부터 1
    grade_post_clear_offset=3,          # N+post_clear 부터 0
    grade_latch=True                    # 퍼블리셔를 latch로 생성할지 여부
)


class HillStopController:
    """
    IMU 피치 + 웨이포인트 인덱스 동시 만족 시 '한 번' 정지 → 정지 유지 → 재가속(버스트)
    또한 위치 기반 플래그 /gps/GRADEUP_ON 을 퍼블리시합니다.
    """

    # ---------- 클래스 초기화 ----------
    def __init__(self):
        # ROS 노드 초기화 (anonymous=True : 노드 이름 충돌 방지용 접미사 추가)'
        #ROS 노드를 초기화합니다.
        #이름: "hill_stop_controller_node"
        #anonymous=True → 만약 같은 이름의 노드가 여러 개 실행되면 뒤에 무작위 숫자를 붙여 충돌을 피합니다.
        rospy.init_node('hill_stop_controller_node', anonymous=True)

        D = DEFAULTS  # 로컬 축약, 기본값 참조

        # ---------- 파라미터(외부에서 바꿀 수 있음) ----------
        # IMU 관련
        self.tilt_threshold_deg    = rospy.get_param('~tilt_threshold_deg',    D['tilt_threshold_deg'])
        self.lpf_alpha             = rospy.get_param('~lpf_alpha',             D['lpf_alpha'])
        self.imu_topic             = rospy.get_param('~imu_topic',             D['imu_topic'])

        # 정지/재개 관련
        # 정지시간
        self.stop_seconds          = float(rospy.get_param('~stop_seconds',          D['stop_seconds']))
        # 재개
        self.resume_speed          = float(rospy.get_param('~resume_speed',          D['resume_speed']))
        # 재개하는 시간
        self.resume_burst_seconds  = float(rospy.get_param('~resume_burst_seconds',  D['resume_burst_seconds']))
        self.speed_topic           = rospy.get_param('~speed_topic',                 D['speed_topic'])

        # 웨이포인트 게이트 관련
        self.wp_index_topic        = rospy.get_param('~wp_index_topic',              D['wp_index_topic'])
        self.target_wp_index       = int(rospy.get_param('~target_waypoint_index',   D['target_waypoint_index']))
        self.require_exact_index   = bool(rospy.get_param('~require_exact_index',    D['require_exact_index']))

        # 운용 옵션
        self.idle_publish          = bool(rospy.get_param('~idle_publish',           D['idle_publish']))
        self.idle_default_speed    = float(rospy.get_param('~idle_default_speed',    D['idle_default_speed']))
        self.debug_log             = bool(rospy.get_param('~debug_log',              D['debug_log']))
        self.force_test_stop       = bool(rospy.get_param('~force_test_stop',        D['force_test_stop']))

        # CSV 로그 경로
        self.csv_path              = rospy.get_param('~csv_path',                    D['csv_path'])

        # GRADEUP_ON 관련 파라미터
        self.grade_enabled             = bool(rospy.get_param('~grade_enabled',            D['grade_enabled']))
        self.grade_topic               = rospy.get_param('~grade_topic',                   D['grade_topic'])
        self.grade_pre_trigger_offset  = int(rospy.get_param('~grade_pre_trigger_offset',  D['grade_pre_trigger_offset']))
        self.grade_post_clear_offset   = int(rospy.get_param('~grade_post_clear_offset',   D['grade_post_clear_offset']))
        self.grade_latch               = bool(rospy.get_param('~grade_latch',              D['grade_latch']))

        # ---------- 내부 상태 변수(초기값) ----------
        self.lpf_pitch_rad = None            # LPF 내부 상태(라디안)
        self.filtered_pitch_deg = 0.0        # LPF 결과(도)

        self.is_stopped = False              # 현재 '정지 유지' 중인지 플래그
        self.stop_start_time = None          # 정지 시작 시각(ros.Time)
        self.stop_duration = rospy.Duration(self.stop_seconds)  # 정지 유지 기간(ros.Duration)
        self.resume_active = False           # 재가속(버스트) 중인지 플래그
        self.resume_end_time = None          # 재가속 종료 시각

        self.pitch_gate_ok = False           # IMU(피치) 게이트 결과
        self.waypoint_gate_ok = False        # 웨이포인트 게이트 결과
        self.current_wp_index = 0            # 마지막 수신된 웨이포인트 인덱스

        # 원샷 래치: 한 번 정지하면 다시 정지하지 않도록 하는 플래그
        self.stop_once_latched = False

        # GRADEUP_ON 현재 상태 (0 또는 1)
        self.grade_on_state = 0

        # ---------- CSV 초기화(속도 기록용) ----------
        self.csv_file = None
        self.csv_writer = None
        self._init_csv()

        # ---------- ROS 퍼블리셔/서브스크라이버 생성 ----------
        # 속도 퍼블리셔 (latch=True로 생성하면 마지막 메시지를 새 구독자가 받음)
        self.vel_pub   = rospy.Publisher(self.speed_topic, Float32, queue_size=10, latch=True)
        # 상태 문자열(디버그)
        self.state_pub = rospy.Publisher('/hill_stop/state', String, queue_size=10, latch=True)
        # LPF된 피치 퍼블리시(디버그)
        self.pitch_pub = rospy.Publisher('/hill_stop/debug_pitch', Float32, queue_size=10, latch=True)
        # 웨이포인트 게이트 여부 퍼블리시(디버그)
        self.wp_ok_pub = rospy.Publisher('/hill_stop/debug_wp_ok', Bool, queue_size=10, latch=True)

        # GRADEUP_ON 퍼블리셔 (latch 여부는 grade_latch 파라미터로 결정)
        self.grade_pub = rospy.Publisher(self.grade_topic, Int32, queue_size=10, latch=self.grade_latch)
        # 초기 상태(0) 시드 퍼블리시 — 구독자가 늦게 붙어도 기본값을 받음
        if self.grade_enabled:
            self.grade_pub.publish(Int32(self.grade_on_state))

        # 구독자: IMU, WP 인덱스, speed(기록용)
        self.imu_sub = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=50)
        self.wp_sub  = rospy.Subscriber(self.wp_index_topic, Int32, self.wp_index_callback, queue_size=10)
        # speed_topic을 구독하여(자신이 퍼블리시한 값 포함) CSV에 기록
        self.speed_sub = rospy.Subscriber(self.speed_topic, Float32, self.speed_callback, queue_size=50)

        # 주기 실행 타이머(0.05s -> 20Hz)
        self.tick_timer = rospy.Timer(rospy.Duration(0.05), self.on_tick)
        # 종료시 CSV 닫기
        rospy.on_shutdown(self._close_csv)

        # 시작 로그(한번 출력)
        rospy.loginfo("[HillStop] started. tilt>=%.1f°, stop=%.1fs, resume=%.1f for %.1fs, gate target wp=%d (exact=%s)",
                      self.tilt_threshold_deg, self.stop_seconds, self.resume_speed,
                      self.resume_burst_seconds, self.target_wp_index, self.require_exact_index)
        rospy.loginfo("[HillStop] IMU: %s, speed topic: %s, wp index topic: %s",
                      self.imu_topic, self.speed_topic, self.wp_index_topic)
        rospy.loginfo("[HillStop] CSV: %s", os.path.abspath(self.csv_path))
        rospy.loginfo("[HillStop] GRADEUP_ON: %s (pre=%d, post_clear=%d, latch=%s, enabled=%s)",
                      self.grade_topic, self.grade_pre_trigger_offset, self.grade_post_clear_offset,
                      self.grade_latch, self.grade_enabled)

        # ------ 테스트용: 시작 직후 강제 1회 정지(옵션) ------
        if self.force_test_stop:
            # 즉시 정지 상태로 진입시키고, 원샷 래치 세팅
            self.is_stopped = True
            self.stop_once_latched = True
            self.stop_start_time = rospy.Time.now()
            self.vel_pub.publish(Float32(0.0))
            self.state_pub.publish(String("STOP(H_FORCE)"))
            rospy.logwarn("[HillStop] force_test_stop=True → 시작 직후 %.1fs 정지", self.stop_seconds)

    # ---------- CSV 헬퍼: 파일 열고 헤더 쓰기 ----------
    def _init_csv(self):
        file_exists = os.path.exists(self.csv_path)
        # append 모드로 열어서 기존 로그 유지
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # 파일이 없거나 비어있으면 헤더 작성
        if not file_exists or os.path.getsize(self.csv_path) == 0:
            self.csv_writer.writerow(['time_sec', 'speed'])
            self.csv_file.flush()

    # ---------- CSV 닫기 ----------
    def _close_csv(self):
        if self.csv_file:
            try: self.csv_file.flush()
            except Exception: pass
            try: self.csv_file.close()
            except Exception: pass

    # ---------- 속도 토픽 콜백(기록 전용) ----------
    def speed_callback(self, msg: Float32):
        # 현재 ROS time(초) 기록
        t = rospy.get_time()
        # CSV에 (time, speed) 기록
        self.csv_writer.writerow([f"{t:.3f}", f"{msg.data:.3f}"])
        self.csv_file.flush()

    # ---------- GRADEUP_ON 업데이트 로직 ----------
    def _update_grade_signal(self):
        """
        current_wp_index가 [target - pre, target + post_clear - 1] 범위면 1, 아니면 0.
        예) target=14, pre=2, post_clear=3 → 12~16에서는 1, 17부터 0
        """
        # 기능이 비활성화 되어 있으면 아무 것도 하지 않음
        if not self.grade_enabled:
            return

        # 안전하게 정수로 변환 후 음수 방지
        pre = max(0, int(self.grade_pre_trigger_offset))    # 2
        post_clear = max(0, int(self.grade_post_clear_offset))  # 3

        # start_idx : 1로 켜줄 시작 인덱스 (음수 방지)
        start_idx = max(0, self.target_wp_index - pre)      # 7 - 2 = 5
        # clear_idx : 이 인덱스부터는 0으로 바꿀 것(반열린 구간)    # 7 + 3 = 10
        clear_idx = self.target_wp_index + post_clear

        # 현재 인덱스가 [start_idx, clear_idx) 범위면 1, 아니면 0
        # 5~10에선 구배토픽 1로 하고 그 범위 밖이면 0
        new_val = 1 if (self.current_wp_index >= start_idx and self.current_wp_index < clear_idx) else 0

        # 값이 바뀔 때만 퍼블리시(네트워크 낭비 방지)
        if new_val != self.grade_on_state:  # self.grade_on_state : 지금까지 퍼블리시된 마지막 상태(0 또는 1)
            self.grade_on_state = new_val
            self.grade_pub.publish(Int32(self.grade_on_state))
            if self.debug_log:
                rospy.loginfo("[HillStop] GRADEUP_ON -> %d (wp=%d, start=%d, clear=%d)",
                              self.grade_on_state, self.current_wp_index, start_idx, clear_idx)

    # ---------- 웨이포인트 인덱스 콜백 ----------
    def wp_index_callback(self, msg: Int32):
        # 외부에서 전달된 인덱스를 내부 상태로 반영
        self.current_wp_index = int(msg.data)
        # require_exact_index 옵션에 따라 게이트 판정 방식이 달라짐
        if self.require_exact_index:
            # 정확히 같은 인덱스일 때만 true
            self.waypoint_gate_ok = (self.current_wp_index == self.target_wp_index)
        else:
            # target 이상이면 true (관대한 판정)
            self.waypoint_gate_ok = (self.current_wp_index >= self.target_wp_index)
        # 디버그 토픽으로 현재 게이트 상태 퍼블리시
        self.wp_ok_pub.publish(Bool(self.waypoint_gate_ok))

        # 웨이포인트가 바뀔 때 GRADEUP_ON도 갱신
        self._update_grade_signal()

    # ---------- IMU 콜백 (쿼터니언 -> 피치 -> LPF -> 게이트) ----------
    def imu_callback(self, msg: Imu):
        # orientation 쿼터니언에서 오일러(roll,pitch,yaw)를 추출
        q = msg.orientation
        (_, pitch_rad, _) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # 내부 규약: 전방 상승을 양수로 만들기 위해 부호 반전
        pitch_rad = -pitch_rad

        # LPF (지수평활) — 첫 입력은 초기값으로 사용
        if self.lpf_pitch_rad is None:
            self.lpf_pitch_rad = pitch_rad
        else:
            # alpha * new + (1-alpha) * prev
            self.lpf_pitch_rad = self.lpf_alpha * pitch_rad + (1.0 - self.lpf_alpha) * self.lpf_pitch_rad

        # LPF 결과를 도 단위로 변환하여 저장/퍼블리시
        self.filtered_pitch_deg = math.degrees(self.lpf_pitch_rad)
        # 피치 게이트 판정 (절대값 기준)
        self.pitch_gate_ok = abs(self.filtered_pitch_deg) >= self.tilt_threshold_deg
        self.pitch_pub.publish(Float32(self.filtered_pitch_deg))

    # ---------- 메인 주기 처리(상태 머신) ----------
    def on_tick(self, _evt):
        # 타이머 콜백에서 호출. _evt는 TimerEvent(사용하지 않음)
        now = rospy.Time.now()

        # 1) 현재 '정지 유지' 중이면 계속 0을 퍼블리시
        if self.is_stopped:
            self.vel_pub.publish(Float32(0.0))
            self.state_pub.publish(String("STOP(HOLD)"))

            # 정지 유지 시간이 경과했으면 재가속 단계로 전환
            if (now - self.stop_start_time) >= self.stop_duration:
                # 정지 해제
                self.is_stopped = False
                self.stop_start_time = None
                # 재가속(버스트) 시작
                self.resume_active = True
                self.resume_end_time = now + rospy.Duration(self.resume_burst_seconds)
                rospy.loginfo("[HillStop] 정지 해제 → 재출발 %.2f for %.1fs",
                              self.resume_speed, self.resume_burst_seconds)

        # 2) 재가속 버스트 중이면 resume_speed 를 퍼블리시
        elif self.resume_active:
            if now < self.resume_end_time:
                self.vel_pub.publish(Float32(self.resume_speed))
                self.state_pub.publish(String("RESUME(BURST)"))
            else:
                # 버스트 종료
                self.resume_active = False
                self.state_pub.publish(String("IDLE"))
                # 이후 속도는 상위 노드가 퍼블리시하도록 설계

        # 3) 정지/재가속 상태가 아닐 때(대기 상태)
        else:
            # 원샷 래치가 비활성이고, 두 게이트가 모두 유효하면 정지 트리거
            if (not self.stop_once_latched) and self.pitch_gate_ok and self.waypoint_gate_ok:
                # 1회성 정지 시작
                self.is_stopped = True
                self.stop_once_latched = True
                self.stop_start_time = now
                rospy.logwarn("[HillStop] 조건 충족! (원샷) pitch=%.2f°, wp_index=%d → %.1fs 정지",
                              self.filtered_pitch_deg, self.current_wp_index, self.stop_seconds)
                self.vel_pub.publish(Float32(0.0))
                self.state_pub.publish(String("STOP(TRIGGER)"))
            else:
                # 조건 미충족 시 IDLE 상태
                self.state_pub.publish(String("IDLE"))
                if self.idle_publish:
                    # keepalive 동작: 평상시에도 속도를 퍼블리시 (상위 노드와 충돌 가능성 주의)
                    self.vel_pub.publish(Float32(self.idle_default_speed))

        # 디버그 로그 (1초 단위로 throttle 처리를 위해 로그스레드 사용)
        if self.debug_log and (rospy.get_time() % 1.0 < 0.05):
            rospy.loginfo_throttle(1.0, "[HillStop] state=%s pitch=%.2f° (gate=%s) wp=%d (gate=%s) latched=%s grade=%d",
                "STOP" if self.is_stopped else ("RESUME" if self.resume_active else "IDLE"),
                self.filtered_pitch_deg, self.pitch_gate_ok, self.current_wp_index,
                self.waypoint_gate_ok, self.stop_once_latched, self.grade_on_state)

    # 노드 실행 (spin)
    def run(self):
        rospy.spin()


# 스크립트 엔트리포인트
if __name__ == '__main__':
    try:
        node = HillStopController()
        node.run()
    except rospy.ROSInterruptException:
        pass
