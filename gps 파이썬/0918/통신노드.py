#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1(Noetic) · RS-485 2포트 송신 노드 (확장)
- /gps/speed_cmd (Float32) → PORT_SPEED 로 [speed(int8), grade_flag(int8)] 2바이트 송신
- /gps/steer_cmd (Float32) → PORT_STEER 로 int8 1바이트 송신
- /gps/GRADEUP_ON (Int32)  → speed 패킷의 두 번째 바이트로 포함
- dSPACE(UART)에서 int8로 읽기 전제
"""

import rospy
from std_msgs.msg import Float32, Int32
import serial
import signal

def clamp_i8(v, limit_abs):
    """Float/Int -> int8 범위(-128..127) 안에서 제한, 먼저 ±limit_abs로 클램프"""
    v = int(round(v))
    if v >  limit_abs: v =  limit_abs
    if v < -limit_abs: v = -limit_abs
    # int8 최종 방어
    if v > 127: v = 127
    if v < -128: v = -128
    return v

def open_serial(port, baud):
    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=0.01)
        rospy.loginfo(f"[RS485] Opened {port} @ {baud}")
        return ser
    except Exception as e:
        rospy.logerr(f"[RS485] Open failed on {port}: {e}")
        return None

class DualPortTx:
    def __init__(self):
        # ── 파라미터 ────────────────────────────────────────────
        self.port_speed = rospy.get_param("~port_speed", "/dev/ttyUSB1")
        self.port_steer = rospy.get_param("~port_steer", "/dev/ttyUSB0")
        self.baud       = int(rospy.get_param("~baud", 9600))
        self.loop_hz    = float(rospy.get_param("~loop_hz", 20))
        self.lim_speed  = int(rospy.get_param("~lim_speed", 100))
        self.lim_steer  = int(rospy.get_param("~lim_steer", 22))
        self.topic_speed= rospy.get_param("~TOPIC_SPEED_CMD", "/gps/speed_cmd")
        self.topic_steer= rospy.get_param("~TOPIC_STEER_CMD", "/gps/steer_cmd")
        self.topic_grade= rospy.get_param("~TOPIC_WP_GRADEUP_ON", "/gps/GRADEUP_ON")

        # ── 상태 값 ─────────────────────────────────────────────
        self.speed_cmd = 0   # Int8
        self.steer_cmd = 0   # Int8
        self.grade_flag = 0  # Int8 (0 or 1)

        # ── 직렬 포트 준비 ─────────────────────────────────────
        self.ser_speed = open_serial(self.port_speed, self.baud)
        self.ser_steer = open_serial(self.port_steer, self.baud)

        # ── 토픽 구독 ──────────────────────────────────────────
        rospy.Subscriber(self.topic_speed, Float32, self.cb_speed, queue_size=20)
        rospy.Subscriber(self.topic_steer, Float32, self.cb_steer, queue_size=20)
        rospy.Subscriber(self.topic_grade, Int32,   self.cb_grade, queue_size=10)

        rospy.loginfo(f"[DualPortTx] speed->{self.port_speed}, steer->{self.port_steer}, baud={self.baud}, hz={self.loop_hz}")
        rospy.loginfo(f"[DualPortTx] topics: {self.topic_speed}, {self.topic_steer}, {self.topic_grade}")

    # 콜백
    def cb_speed(self, msg: Float32):
        self.speed_cmd = clamp_i8(msg.data * 10, self.lim_speed)

    def cb_steer(self, msg: Float32):
        self.steer_cmd = clamp_i8(msg.data, self.lim_steer)

    def cb_grade(self, msg: Int32):
        self.grade_flag = 1 if msg.data != 0 else 0

    # 직렬 재오픈 헬퍼
    def _ensure_ports(self):
        if self.ser_speed is None or (not self.ser_speed.is_open):
            self.ser_speed = open_serial(self.port_speed, self.baud)
        if self.ser_steer is None or (not self.ser_steer.is_open):
            self.ser_steer = open_serial(self.port_steer, self.baud)

    # 메인 루프
    def spin(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            try:
                self._ensure_ports()

                # speed → PORT_SPEED (2바이트: [speed, grade_flag])
                if self.ser_speed and self.ser_speed.is_open:
                    packet = bytearray(2)
                    packet[0] = int(self.speed_cmd).to_bytes(1, "little", signed=True)[0]
                    packet[1] = int(self.grade_flag).to_bytes(1, "little", signed=True)[0]
                    self.ser_speed.write(packet)
                else:
                    rospy.logwarn_throttle(2.0, "[RS485] speed port not open")

                # steer → PORT_STEER (1바이트, signed=True)
                if self.ser_steer and self.ser_steer.is_open:
                    self.ser_steer.write(int(self.steer_cmd).to_bytes(1, "little", signed=True))
                else:
                    rospy.logwarn_throttle(2.0, "[RS485] steer port not open")

            except Exception as e:
                rospy.logerr_throttle(1.0, f"[DualPortTx] TX error: {e}")

            rate.sleep()

def main():
    rospy.init_node("gps_to_rs485_dualport_node")

    node = DualPortTx()

    # 안전 종료 처리
    def _sigint(_sig, _frm):
        rospy.signal_shutdown("SIGINT")
        try:
            if node.ser_speed: node.ser_speed.close()
            if node.ser_steer: node.ser_steer.close()
        except Exception:
            pass
    signal.signal(signal.SIGINT, _sigint)

    node.spin()

if __name__ == "__main__":
    main()
