import argparse
from fabric import Connection
import time
 
# argparse를 통해 IP 주소 입력 받기
parser = argparse.ArgumentParser(description='IP 주소를 입력받아 원격 명령을 실행합니다.')
parser.add_argument('--ip', type=str, required=True, help='제어할 원격 장치의 IP 주소')
args = parser.parse_args()
 
# SSH 연결 설정
c1 = Connection(host=args.ip, user='jetson', connect_kwargs={'password': 'yahboom'})
 
try:
    c1.run('python3 /home/jetson/Desktop/demo/jetsonServer/app/cameracontrol/ptcontrol.py -p 0 -t 0')
    time.sleep(5)

except KeyboardInterrupt:
    print("\n프로그램이 중단되었습니다.")
finally:
    print("종료 완료.")