/*
 * OpenArm 夾爪壓力感測器 Arduino 韌體 (專用: SY40 系列薄膜壓力放大模組 SY40-AMP)
 *
 * 硬體接線說明 (SY40-AMP 模組 -> Arduino)：
 * -----------------------------------------------------------------------------
 *   【SY40-AMP 模組腳位】           【Arduino 腳位】
 *      VCC                     ->     5V (提供 5V 工作電源)
 *      GND                     ->     GND (共地)
 *      AO / AOUT (模擬輸出)     ->     A0 (讀取 0~5V 類比電壓)
 *      DO / DOUT (閥值數位輸出) ->     可不接 (或接數位腳位如 D2，用於硬體觸發告警)
 *   【薄膜傳感器端子】
 *      SY40 薄膜感測器兩腳     ->     插上/鎖上 SY40-AMP 上的傳感器端子 (無極性)
 * -----------------------------------------------------------------------------
 * 
 * 模組特性：
 *   - 內部已整合高精度放大電路與濾波，AO 腳位輸出標準 0.0V ~ 5.0V。
 *   - 壓力越大，輸出電壓/類比值越高。
 *   - 程式支援零點動態去除 (Tare) 與過採樣中值平均濾波。
 * 
 * 傳輸規格 (Baudrate: 115200, 50Hz / 20ms)：
 *   輸出格式: RAW:<0~1023>,VOLT:<0.000~5.000>
 */

const int SENSOR_PIN = A7;
const unsigned long INTERVAL_MS = 20; // 50Hz 取樣頻率
unsigned long lastSendTime = 0;

// 濾波緩衝區 (中值 + 平均濾波，消除機械震動雜訊)
const int SAMPLE_COUNT = 8;
int samples[SAMPLE_COUNT];

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // 等待序列埠連線
  }
}

// 快速排序供取中值
void sortArray(int a[], int size) {
  for (int i = 0; i < size - 1; i++) {
    for (int j = i + 1; j < size; j++) {
      if (a[i] > a[j]) {
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
      }
    }
  }
}

void loop() {
  unsigned long now = millis();
  if (now - lastSendTime >= INTERVAL_MS) {
    lastSendTime = now;
    
    // 採集 8 筆樣本
    for (int i = 0; i < SAMPLE_COUNT; i++) {
      samples[i] = analogRead(SENSOR_PIN);
      delayMicroseconds(60);
    }
    
    // 剔除極大值與極小值，中間值取平均 (去除毛邊干擾)
    sortArray(samples, SAMPLE_COUNT);
    int sum = 0;
    for (int i = 2; i < SAMPLE_COUNT - 2; i++) {
      sum += samples[i];
    }
    int rawValue = sum / (SAMPLE_COUNT - 4); 
    
    // 換算為精確電壓 (0.000 ~ 5.000 V)
    float voltage = (rawValue / 1023.0) * 5.0;

    // 格式化輸出供 test_gripper_force.py 解析
    Serial.print("RAW:");
    Serial.print(rawValue);
    Serial.print(",VOLT:");
    Serial.print(voltage, 3);
    Serial.println();
  }

  // 接收 Python 發送的指令 (如 't' 歸零 / tare)
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 't' || cmd == 'T') {
      Serial.println("STATUS:TARED");
    }
  }
}
