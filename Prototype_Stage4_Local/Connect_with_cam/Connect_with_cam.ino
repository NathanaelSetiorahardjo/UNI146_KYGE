#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>

// ====== CAMERA MODEL ======
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// ====== OLED DISPLAY SETTINGS ======
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ====== WIFI CREDENTIALS ======
const char* ssid = "YONG";
const char* password = "guramebakar1912";

// ====== UBIDOTS MQTT SETTINGS ======
const char* mqtt_server = "industrial.api.ubidots.com";
const int mqtt_port = 1883;
const char* mqtt_token = "BBUS-nVBcjWjcD0R6gTWTlf2UxAeGMRHO5I";
const char* mqtt_device_label = "prototype";
const char* mqtt_topic_prediction = "/v1.6/devices/prototype/prediction";
const char* mqtt_topic_confidence = "/v1.6/devices/prototype/confidence";

// MQTT + WiFi clients
WiFiClient espClient;
PubSubClient client(espClient);

// Global vars to store latest values
char currentLetter = '-';
float currentConfidence = 0.0;

// ====== SETUP WIFI FUNCTION ======
void setup_wifi() {
  delay(10);
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// ====== MQTT CALLBACK FUNCTION ======
void callback(char* topic, byte* payload, unsigned int length) {
  String message;
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  Serial.println(message);

  StaticJsonDocument<64> doc;
  DeserializationError error = deserializeJson(doc, message);
  if (error) {
    Serial.print("deserializeJson() failed: ");
    Serial.println(error.c_str());
    return;
  }

  float value = doc["value"];
  Serial.print("Parsed value: ");
  Serial.println(value);

  if (String(topic) == mqtt_topic_prediction) {
    int letterCode = (int)value;
    if (letterCode >= 0 && letterCode <= 25) {
      currentLetter = 'A' + letterCode;
    } else {
      currentLetter = '?';
    }
  } else if (String(topic) == mqtt_topic_confidence) {
    currentConfidence = value;
  }

  // Update OLED display
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 10);
  display.print("Letter: ");
  display.println(currentLetter);
  display.setCursor(0, 40);
  display.print("Con: ");
  display.print(currentConfidence * 100, 1);
  display.println("%");
  display.display();
}


// ====== MQTT RECONNECT FUNCTION ======
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP32CAMClient", mqtt_token, "")) {
      Serial.println("connected");
      client.subscribe(mqtt_topic_prediction);
      client.subscribe(mqtt_topic_confidence);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

// ====== START CAMERA SERVER FUNCTION ======
void startCameraServer(); // Forward declaration

// ====== SETUP FUNCTION ======
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println("Booting...");

  // Force I2C pins to match wiring
  Wire.begin(14, 15);  // SDA=14, SCL=15

  // Initialize OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for (;;);
  }
  display.clearDisplay();
  display.display();

  // Connect to Wi-Fi
  setup_wifi();

  // MQTT setup
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  // Initialize Camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if(psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // Start streaming server
  startCameraServer();
  Serial.println("Camera ready! Stream at: http://" + WiFi.localIP().toString());
}

// ====== LOOP FUNCTION ======
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
