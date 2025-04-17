#ifndef APP_HTTPD_H
#define APP_HTTPD_H

#include "esp_http_server.h"
#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"

typedef struct {
    httpd_handle_t stream_httpd;
    httpd_handle_t camera_httpd;
} httpd_inst_t;

void startCameraServer();
void setupLedFlash(int pin);

#endif