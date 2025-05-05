import cv2
import numpy as np
from picamera2 import Picamera2
import time
import serial
import struct
import math
import ncnn  # NCNN Python binding
import re


# Đường dẫn model NCNN cho phát hiện biển báo giao thông
PARAM_PATH     = "/home/labthayphuc/Desktop/yolov11ncnn/yolov11n_int8_final.param"
BIN_PATH       = "/home/labthayphuc/Desktop/yolov11ncnn/yolov11n_int8_final.bin"
# Thông số NCNN
NCNN_INPUT_SIZE           = 640
NCNN_INPUT_NAME           = "in0"
NCNN_OUTPUT_NAME          = "out0"
NCNN_MEAN_VALS            = [0.0, 0.0, 0.0]
NCNN_NORM_VALS            = [1/255.0, 1/255.0, 1/255.0]
NCNN_CONFIDENCE_THRESHOLD = 0.35
NCNN_NMS_THRESHOLD        = 0.35
NCNN_NUM_CLASSES          = 19
NCNN_CLASS_NAMES = [
    "Pedestrian Crossing", "Radar", "Speed Limit -100-", "Speed Limit -120-",
    "Speed Limit -20-", "Speed Limit -30-", "Speed Limit -40-", "Speed Limit -50-",
    "Speed Limit -60-", "Speed Limit -70-", "Speed Limit -80-", "Speed Limit -90-",
    "Stop Sign", "Traffic Light -Green-", "Traffic Light -Off-",
    "Traffic Light -Red-", "Traffic Light -Yellow-", "Person", "Car"
]

# Cấu hình camera và serial
SERIAL_PORT     = '/dev/ttyUSB0'      # Cổng nối ESP32
BAUD_RATE       = 115200              # Tốc độ baud
PICAM_SIZE      = (640, 480)          # Kích thước ảnh từ PiCamera
PICAM_FRAMERATE = 30                  # FPS của camera

# Warp phối cảnh
leftTop, heightTop, rightTop       = 160, 150, 160
leftBottom, heightBottom, rightBottom = 20, 380, 20

# Thông số điều khiển tốc độ
DEFAULT_MAX_SPEED       = 120         # km/h
MANUAL_MAX_SPEED        = 100         # km/h khi điều khiển tay
MANUAL_ACCELERATION     = 5           # km/h mỗi nhấn phím
AUTO_MODE_SPEED_STRAIGHT = 30         # km/h khi đường thẳng
AUTO_MODE_SPEED_CURVE    = 20         # km/h khi vào cua
MIN_SIGN_SIZE            = 100        # px tối thiểu để xét biển

# --- Globals ---
frame = None
imgWarp = None
net = None
ser = None
speed_motor_requested = 0            # km/h do người điều khiển yêu cầu
current_speed_limit    = DEFAULT_MAX_SPEED
cte_f = 0.0                          # Cross-Track Error
prev_time = 0.0                      # thời điểm vòng trước
# Biến lane
pre_diff = 0.0
lane_width = lane_width_max = 500
left_point = right_point = -1
interested_line_y = 0

# Tham số EMA smoothing (nội tuyến)
prev_cte_ema = 0.0    # Giá trị EMA CTE của khung trước
alpha_cte = 0.5      # Hệ số làm mịn EMA (0 < alpha_cte <= 1)



# --- PID Class ---
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.integrator_min = -50.0
        self.integrator_max = 50.0
        self.alpha = 0.1  # lọc IIR cho D

    def update(self, error, dt):
        """Tính P, I, D và trả về (output, góc servo)"""
        if dt <= 0: dt = 1e-3
        
        # tính P
        p = self.kp * error
        # tính I
        self.integral += error * dt
        self.integral = max(self.integrator_min, min(self.integral, self.integrator_max))
        i = self.ki * self.integral
        # tính D
        derivative = (error - self.prev_error) / dt
        derivative_filtered = self.alpha * derivative + (1 - self.alpha) * self.prev_derivative #làm mượt bằng EMA
        d = self.kd * derivative_filtered
        # cập nhật trạng thái
        self.prev_error = error
        self.prev_derivative = derivative
        # tổng output và góc servo (90 trung tâm)
        output = round(p + i + d)
        servo_angle = 90 + output
        
        return output ,servo_angle

# --- Khởi tạo Serial ---
print(f"Initializing Serial on {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Serial initialized at {BAUD_RATE} baud.")
except Exception as e:
    print(f"Warning: Serial unavailable: {e}")
    ser = None

# --- Load NCNN Model ---
net = ncnn.Net()
net.opt.use_vulkan_compute = False
net.opt.num_threads = 4
net.load_param(PARAM_PATH)
net.load_model(BIN_PATH)

# --- Khởi tạo PiCamera2 ---
piCam = Picamera2()
piCam.preview_configuration.main.size = PICAM_SIZE
piCam.preview_configuration.main.format = 'RGB888'
piCam.preview_configuration.controls.FrameRate = float(PICAM_FRAMERATE)
piCam.preview_configuration.align()
piCam.configure('preview')
piCam.start()
time.sleep(1.0)

# --- Helper Functions ---
def process_image():
    """- Đọc ảnh, chuyển sang BGR, Canny, warp phối cảnh"""
    global frame, imgWarp
    try:
        rgb = piCam.capture_array()
    except:
        return False
    if rgb is None:
        return False
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11,11), 0)
    edges = cv2.Canny(blur, 70, 120)
    h, w = edges.shape
    pts1 = np.float32([(leftTop, heightTop), (w-rightTop, heightTop), (leftBottom, heightBottom), (w-rightBottom, heightBottom)])
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(edges, M, (w,h))
    contours, hierarchy = cv2.findContours(imgWarp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgWarp, contours, -1, (255, 255, 255), 3) 
    lines = cv2.HoughLinesP(imgWarp, 1, np.pi/180, threshold=150, minLineLength=70, maxLineGap=0)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(imgWarp, (x1, y1), (x2, y2), (255, 255, 255), 3)  
    return True


def ve_truc_anh(img, step=50):
    """Vẽ trục và lưới debug"""
    h, w = img.shape[:2]
    viz = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.line(viz, (0,h-1), (w,h-1), (200,200,200), 1)
    cv2.line(viz, (0,0), (0,h), (200,200,200), 1)
    for x in range(0, w, step): cv2.line(viz, (x,h-10),(x,h),(200,200,200),1)
    for y in range(0, h, step): cv2.line(viz, (0,y),(10,y),(200,200,200),1)
    return viz


def SteeringAngle():
    """Tính CTE bằng quét hàng ngang"""
    global cte_f, pre_diff, lane_width, lane_width_max, left_point, right_point, interested_line_y, prev_cte_ema, alpha_cte

    prev_cte_ema = cte_f
    pre_diff = 0
    h, w = imgWarp.shape
    left_point = right_point = -1
    center = w//2
    set_y = 401
    step = -5
    u = (h-set_y)/abs(step)
    ki = (-u/h) + 1
    for i in range(h-1, set_y, step):
        interested_line_y = i
        row = imgWarp[i]
        for x in range(center, -1, -1):
            if row[x]>0: left_point=x; break
        for x in range(center+1, w):
            if row[x]>0: right_point=x; break
        if left_point!=-1 and right_point!=-1: lane_width = right_point-left_point
        if left_point!=-1 and right_point==-1: right_point=left_point+lane_width_max
        if right_point!=-1 and left_point==-1: left_point=right_point-lane_width_max
        mid = (left_point+right_point)/2
        diff = center-mid
        diff = diff*ki + pre_diff
        pre_diff = diff        
    raw_cte = pre_diff 
    # Lọc EMA qua các khung để mượt CTE
    cte_filter = alpha_cte * raw_cte + (1.0 - alpha_cte) * prev_cte_ema
    cte_rad = math.atan((cte_filter-center) / (set_y))
    cte_deg = math.degrees(cte_rad)
    cte_f = round(cte_deg*0.45)

def signal_motor(key):
    """Điều khiển tốc độ qua phím w/s/a/x"""
    global speed_motor_requested
    if key == ord('w'): speed_motor_requested = min(speed_motor_requested+MANUAL_ACCELERATION, MANUAL_MAX_SPEED)
    if key == ord('s'): speed_motor_requested = max(speed_motor_requested-MANUAL_ACCELERATION, 0)
    if key == ord('a'): speed_motor_requested = AUTO_MODE_SPEED_STRAIGHT if abs(cte_f)<10 else AUTO_MODE_SPEED_CURVE
    if key == ord('x'): speed_motor_requested = 0


def detect_signs_and_get_results(frame_bgr):
    """Phát hiện biển báo và NMS"""
    results=[]
    if frame_bgr is None: return results
    h,w = frame_bgr.shape[:2]
    pad = np.full((NCNN_INPUT_SIZE,NCNN_INPUT_SIZE,3),114,np.uint8)
    scale=min(NCNN_INPUT_SIZE/w,NCNN_INPUT_SIZE/h)
    nw,nh=int(w*scale),int(h*scale)
    dw,dh=(NCNN_INPUT_SIZE-nw)//2,(NCNN_INPUT_SIZE-nh)//2
    pad[dh:dh+nh, dw:dw+nw]=frame_bgr
    mat=ncnn.Mat.from_pixels(pad,ncnn.Mat.PixelType.PIXEL_BGR,NCNN_INPUT_SIZE,NCNN_INPUT_SIZE)
    mat.substract_mean_normalize(NCNN_MEAN_VALS,NCNN_NORM_VALS)
    ex=net.create_extractor(); ex.input(NCNN_INPUT_NAME, mat)
    _,out=ex.extract(NCNN_OUTPUT_NAME)
    data=np.array(out)
    if data.ndim==3: data=data[0]
    if data.ndim==2 and data.shape[0]==NCNN_NUM_CLASSES+4: data=data.T
    if data.ndim!=2: return results
    boxes,confs,cls=[],[],[]
    for det in data:
        scores=det[4:]; conf=scores.max(); cid=scores.argmax()
        if conf<NCNN_CONFIDENCE_THRESHOLD: continue
        cx,cy,wn,hn=det[:4]
        x1,x2=cx-wn/2,cx+wn/2; y1,y2=cy-hn/2,cy+hn/2
        x1=max(0,(x1-dw)/scale); y1=max(0,(y1-dh)/scale)
        x2=min(w,(x2-dw)/scale); y2=min(h,(y2-dh)/scale)
        boxes.append([int(x1),int(y1),int(x2-x1),int(y2-y1)])
        confs.append(float(conf)); cls.append(int(cid))
    inds=cv2.dnn.NMSBoxes(boxes,confs,NCNN_CONFIDENCE_THRESHOLD,NCNN_NMS_THRESHOLD)
    for i in (inds.flatten() if hasattr(inds,'flatten') else inds):
        name=NCNN_CLASS_NAMES[cls[i]] if cls[i]<len(NCNN_CLASS_NAMES) else f"ID:{cls[i]}"
        x,y,w0,h0=boxes[i]
        results.append({"name":name,"confidence":confs[i],"box":[x,y,w0,h0],"width":w0,"height":h0})
    return results


def parse_speed_limit(name):
    """Lấy giá trị số từ tên lớp 'Speed Limit -X-'"""
    m=re.search(r"Speed Limit -(\d+)-",name)
    return int(m.group(1)) if m else None


def transform_speed(vel):
    """Chuyển km/h -> RPM động cơ"""
    f=vel/10
    rpm=(f*30*2.85)/(3.14*0.0475*3.6)
    return int(round(rpm))

# --- Main Loop ---
def main():
    """Vòng lặp chính: xử lý ảnh, PID lái, gửi lệnh, debug"""
    global prev_time,cte_f,current_speed_limit
    pid1=PID(0.35,0.0008,0.25)
    pid2=PID(0.25,0.0005,0.3)
    pid3=PID(0.15,0.0002,0.1)
    prev_time=time.time()
    try:
        while True:
            now=time.time(); dt=now-prev_time if now!=prev_time else 1e-3; prev_time=now
            if not process_image(): time.sleep(0.01); continue
            SteeringAngle()
            signs=detect_signs_and_get_results(frame)
            stop=False; seen=False; new_lim=-1
            for s in signs:
                if s['width']>MIN_SIGN_SIZE or s['height']>MIN_SIGN_SIZE:
                    if s['name'] in ['Stop Sign','Traffic Light -Red-']: stop=True
                    lim=parse_speed_limit(s['name'])
                    if lim and (not seen or lim<new_lim): new_lim=lim; seen=True
            if seen and new_lim!=current_speed_limit: current_speed_limit=new_lim
            vel_req=min(speed_motor_requested,current_speed_limit)
            if stop: vel_req=0
            rpm_cmd=transform_speed(vel_req)
            if vel_req<=20: _,steer=pid1.update(cte_f,dt)
            elif vel_req<=35: _,steer=pid2.update(cte_f, dt)
            else: _,steer=pid3.update(cte_f, dt)    
            print(f"CTE: {cte_f:.2f}, Speed: {vel_req} km/h, Steering: {steer}, Signs: {signs}")
            
            
            if ser:
                data=struct.pack('<ii',steer,rpm_cmd)
                ser.write(b'<'+data+b'>'); ser.flush()
            if imgWarp is not None:
                cv2.imshow('Warp',ve_truc_anh(imgWarp))
                #cv2.imshow('frame', frame)
            key=cv2.waitKey(1)&0xFF; signal_motor(key)
            if key==ord('q'): break
    except KeyboardInterrupt:
        pass
    finally:
        if ser and ser.is_open: ser.write(b'<' + struct.pack('<ii',90,0) + b'>'); ser.close()
        piCam.stop(); cv2.destroyAllWindows()

if __name__=='__main__':
    main()
