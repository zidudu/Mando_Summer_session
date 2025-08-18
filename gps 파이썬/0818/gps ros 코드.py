#! /usr/bin/env python3
import rospy
import math
import csv
from std_msgs.msg import Float32, String
from sensor_msgs.msg import NavSatFix
from ublox_msgs.msg import NavRELPOSNED, NavPVT  # 필요한 메시지 임포트

# =============== (옵션) ROS 래퍼 ===============
# waypoint_tracker_node.py
# 필요할 때만 사용: import 실패해도 코어 사용에는 영향 없음
try:
    _HAVE_RELPOSNED = True
except Exception:
    _HAVE_RELPOSNED = False

try:
    _HAVE_NAVPVT = True
except Exception:
    _HAVE_NAVPVT = False

_ROS_OK = True


# ================== TrackerCore 클래스 ==================

class TrackerCore:
    def __init__(self, csv_path, wheelbase, look_dist0, k_ld, k_v, arrive_r, exit_r, const_speed, log_csv=""):
        self.csv_path = csv_path
        self.wheelbase = wheelbase
        self.look_dist0 = look_dist0
        self.k_ld = k_ld
        self.k_v = k_v
        self.arrive_r = arrive_r
        self.exit_r = exit_r
        self.const_speed = const_speed
        self.log_csv = log_csv
        self.waypoints = self.load_waypoints(csv_path)  # CSV 파일에서 경로 로딩

    def load_waypoints(self, path):
        waypoints = []
        with open(path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 첫 번째 줄(헤더)을 건너뜁니다.
            for row in reader:
                waypoints.append((float(row[0]), float(row[1])))  # x, y 좌표로 저장
        return waypoints

    def update(self, latitude, longitude, now, rtk_status):
        closest_wp = self.get_closest_waypoint(latitude, longitude)
        dx = closest_wp[0] - latitude
        dy = closest_wp[1] - longitude
        distance = (dx**2 + dy**2) ** 0.5
        speed = self.const_speed if distance > self.arrive_r else 0
        steer = self.calculate_steering_angle(dx, dy)

        if rtk_status == 2:
            rtk_tf = "FIX"
        else:
            rtk_tf = "FLOAT"

        return speed, steer, rtk_tf, distance

    def get_closest_waypoint(self, lat, lon):
        closest_wp = min(self.waypoints, key=lambda wp: (wp[0] - lat)**2 + (wp[1] - lon)**2)
        return closest_wp

    def calculate_steering_angle(self, dx, dy):
        return math.atan2(dy, dx) * 180 / math.pi  # 라디안을 각도로 변환

    def save_logs(self):
        pass


# ================== TrackerRosNode 클래스 ==================

class TrackerRosNode:
    def __init__(self):
        # 파라미터
        csv_path = rospy.get_param("~csv_path", "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/waypoints_example.csv")
        wheelbase  = float(rospy.get_param("~wheelbase", 0.28))
        look_dist0 = float(rospy.get_param("~look_dist0", 1.2))
        k_ld       = float(rospy.get_param("~k_ld", 0.3))
        k_v        = float(rospy.get_param("~k_v", 0.5))
        arrive_r   = float(rospy.get_param("~arrive_r", 0.5))
        exit_r     = float(rospy.get_param("~exit_r", 0.8))
        const_speed= float(rospy.get_param("~const_speed", 1.0))
        self.do_publish = bool(rospy.get_param("~do_publish", True))

        # 토픽 이름(네임스페이스 변경 시 여기만 바꾸면 됨)
        ublox_ns      = rospy.get_param("~ublox_ns", "/ublox")
        self.fix_topic     = rospy.get_param("~fix_topic",      ublox_ns + "/fix")
        self.relpos_topic  = rospy.get_param("~relpos_topic",   ublox_ns + "/navrelposned")
        self.navpvt_topic  = rospy.get_param("~navpvt_topic",   ublox_ns + "/navpvt")

        # TrackerCore 인스턴스 생성
        self.core = TrackerCore(csv_path=csv_path, wheelbase=wheelbase, look_dist0=look_dist0,
                                k_ld=k_ld, k_v=k_v, arrive_r=arrive_r, exit_r=exit_r,
                                const_speed=const_speed, log_csv=rospy.get_param("~log_csv", ""))
        
        # 속도, 조향각, 상태 보내기
        self.pub_speed = rospy.Publisher("/vehicle/speed_cmd", Float32, queue_size=10)
        self.pub_steer = rospy.Publisher("/vehicle/steer_cmd", Float32, queue_size=10)
        self.pub_rtk   = rospy.Publisher("/rtk/status", String, queue_size=10)

        # 최근 RTK 상태를 carrSoln 유사 코드(0/1/2)로 유지
        self._last_carr = 0

        # 구독: fix는 필수
        rospy.Subscriber(self.fix_topic, NavSatFix, self._cb_fix, queue_size=100)

        # RELPOSNED가 import 됐으면 구독 시도
        if _HAVE_RELPOSNED:
            rospy.Subscriber(self.relpos_topic, NavRELPOSNED, self._cb_relpos, queue_size=50)

        # NAV-PVT도 가능하면 구독
        if _HAVE_NAVPVT:
            rospy.Subscriber(self.navpvt_topic, NavPVT, self._cb_navpvt, queue_size=50)

        rospy.loginfo("[tracker_ros] ready: fix=%s relpos=%s navpvt=%s",
                      self.fix_topic, self.relpos_topic if _HAVE_RELPOSNED else "N/A",
                      self.navpvt_topic if _HAVE_NAVPVT else "N/A")

    def _cb_relpos(self, msg: 'NavRELPOSNED'):
        carr_bits = int(msg.flags) & int(NavRELPOSNED.FLAGS_CARR_SOLN_MASK)
        if carr_bits == int(NavRELPOSNED.FLAGS_CARR_SOLN_FIXED):
            self._last_carr = 2
        elif carr_bits == int(NavRELPOSNED.FLAGS_CARR_SOLN_FLOAT):
            self._last_carr = 1
        else:
            self._last_carr = 0

    def _cb_navpvt(self, msg: 'NavPVT'):
        phase = int(msg.flags) & int(NavPVT.FLAGS_CARRIER_PHASE_MASK)
        if phase == int(NavPVT.CARRIER_PHASE_FIXED):
            self._last_carr = 2
        elif phase == int(NavPVT.CARRIER_PHASE_FLOAT):
            self._last_carr = 1
        else:
            self._last_carr = 0

    def _cb_fix(self, msg: 'NavSatFix'):
        if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
            return
        now = rospy.Time.now().to_sec()
        v_cmd, steer_deg, rtk_tf, _ = self.core.update(
            msg.latitude, msg.longitude, now, self._last_carr
        )

        # 위도, 경도, RTK 상태 출력
        rospy.loginfo(f"Latitude: {msg.latitude}, Longitude: {msg.longitude}, RTK Status: {rtk_tf}")

        # 계산된 값 출력
        rospy.loginfo(f"Calculated Speed: {v_cmd}, Steering Angle: {steer_deg}, RTK Status: {rtk_tf}")

        if self.do_publish:
            self.pub_speed.publish(Float32(data=float(v_cmd)))
            self.pub_steer.publish(Float32(data=float(steer_deg)))
            self.pub_rtk.publish(String(data=rtk_tf))

    def spin(self):
        rate = rospy.Rate(200)
        while not rospy.is_shutdown():
            rate.sleep()
        self.core.save_logs()


def _run_ros_node():
    rospy.init_node("waypoint_tracker_returnable")
    node = TrackerRosNode()
    try:
        node.spin()
    finally:
        node.core.save_logs()


if __name__ == "__main__":
    if not _ROS_OK:
        raise RuntimeError(
            "ROS dependencies missing. Install ros-noetic-ublox / ros-noetic-ublox-msgs "
            "and source your setup.bash"
        )
    _run_ros_node()
