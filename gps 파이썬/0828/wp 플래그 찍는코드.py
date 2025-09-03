#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 기존 waypoint_tracker_topics.py에서 가져온 함수 ---
def latlon_to_meters(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y
# ----------------------------------------------------

class WaypointEditor:
    def __init__(self, csv_path):
        """에디터 초기화 및 실행"""
        self.csv_path = csv_path
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"'{self.csv_path}' 파일을 찾을 수 없습니다.")

        self.df = None
        self.selected_index = None

        # 데이터 로드 및 미터 좌표 변환
        self.load_data()

        # Matplotlib 시각화 설정
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # 초기 플롯 그리기
        self.plot_waypoints()
        
        print("--- 웨이포인트 에디터 ---")
        print("1. 그래프에서 Action을 추가할 지점을 마우스로 클릭하세요.")
        print("2. 터미널 안내에 따라 Action 이름을 입력하고 Enter를 누르세요.")
        print("3. 원하는 만큼 반복 후, 그래프 창을 활성화하고 's' 키를 눌러 저장하세요.")

    def load_data(self):
        """CSV 파일을 로드하고 x, y 미터 좌표를 계산합니다."""
        self.df = pd.read_csv(self.csv_path)

        # 'Action' 열이 없으면 새로 생성
        if 'Action' not in self.df.columns:
            self.df['Action'] = ""
        self.df['Action'] = self.df['Action'].fillna("") # NaN 값을 빈 문자열로 변경

        # 위도, 경도를 미터 좌표로 변환하여 새로운 열에 저장
        coords = [latlon_to_meters(row['Lat'], row['Lon']) for _, row in self.df.iterrows()]
        self.df['x'] = [c[0] for c in coords]
        self.df['y'] = [c[1] for c in coords]

    def plot_waypoints(self):
        """현재 웨이포인트 데이터를 그래프에 그립니다."""
        self.ax.clear()
        
        # 모든 웨이포인트 그리기
        self.ax.plot(self.df['x'], self.df['y'], '.-', color='skyblue', label='Path', zorder=1)

        # Action이 있는 웨이포인트 강조 및 텍스트 표시
        action_points = self.df[self.df['Action'] != ""]
        if not action_points.empty:
            self.ax.scatter(action_points['x'], action_points['y'],
                            color='limegreen', s=100, label='With Action', zorder=2)
            for i, row in action_points.iterrows():
                self.ax.text(row['x'], row['y'] + 0.5, row['Action'],
                             color='green', ha='center', fontsize=9, style='italic')

        # 현재 선택된 웨이포인트 강조
        if self.selected_index is not None:
            selected_pt = self.df.loc[self.selected_index]
            self.ax.scatter(selected_pt['x'], selected_pt['y'],
                            color='red', marker='x', s=150, linewidth=3,
                            label='Selected', zorder=3)

        self.ax.set_title("Waypoint Editor | Click to Select | Press 's' to Save")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.axis('equal')
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.legend()
        self.fig.canvas.draw()

    def on_click(self, event):
        """마우스 클릭 이벤트를 처리합니다."""
        # 그래프 안에서 클릭했는지 확인
        if event.inaxes != self.ax:
            return

        click_x, click_y = event.xdata, event.ydata

        # 클릭 지점과 가장 가까운 웨이포인트 찾기
        distances = np.hypot(self.df['x'] - click_x, self.df['y'] - click_y)
        self.selected_index = distances.idxmin()

        print(f"\n> {self.selected_index}번 웨이포인트가 선택되었습니다.")
        
        # 선택된 포인트를 반영하여 다시 그리기
        self.plot_waypoints()
        
        # 터미널에서 사용자 입력 받기
        self.prompt_for_action()

    def prompt_for_action(self):
        """터미널에서 Action 이름을 입력받아 DataFrame을 업데이트합니다."""
        if self.selected_index is None:
            return

        current_action = self.df.loc[self.selected_index, 'Action']
        prompt = f"  ㄴ 현재 Action: '{current_action}'. 변경할 Action 이름을 입력하세요 (삭제는 Enter): "
        
        new_action = input(prompt).strip()
        self.df.loc[self.selected_index, 'Action'] = new_action

        print(f"  -> Action이 '{new_action}'으로 업데이트 되었습니다. ('s'를 눌러 저장하세요)")

        # Action 텍스트가 변경되었을 수 있으므로 다시 그리기
        self.plot_waypoints()

    def on_key_press(self, event):
        """키보드 입력 이벤트를 처리합니다."""
        if event.key == 's':
            self.save_data()

    def save_data(self):
        """변경 사항을 CSV 파일에 저장합니다."""
        # 저장 전 원본 파일을 백업
        backup_path = self.csv_path + '.bak'
        print(f"\n> 원본 파일을 '{backup_path}'에 백업합니다.")
        os.rename(self.csv_path, backup_path)

        # 필요한 열만 선택하여 저장
        save_df = self.df[['Lat', 'Lon', 'Action']]
        save_df.to_csv(self.csv_path, index=False)
        
        print(f"> 변경 사항이 '{self.csv_path}'에 성공적으로 저장되었습니다.")
        plt.title("SAVED! | Click to Select | Press 's' to Save")
        self.fig.canvas.draw()

    def run(self):
        """에디터를 실행하고 창을 띄웁니다."""
        plt.show()


if __name__ == '__main__':
    # --- 설정 ---
    # rospkg가 없으므로 경로를 직접 지정해줍니다.
    # 사용자의 환경에 맞게 이 경로를 수정해주세요.
    CSV_FILE_PATH = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker/config/left_lane.csv')
    
    try:
        editor = WaypointEditor(CSV_FILE_PATH)
        editor.run()
    except FileNotFoundError as e:
        print(f"[오류] {e}")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")
