#include <Adafruit_ADS1X15.h>

const unsigned long SAMPLING_FREQ_HZ = 2000;             // desired sampling frequency in Hz
const unsigned long SAMPLING_INTERVAL_US = 1000000UL / SAMPLING_FREQ_HZ;   // interval between samples in microseconds

Adafruit_ADS1115 ads;  /* Use this for the 16-bit version */
unsigned long previousMicros = 0;

void setup(void)
{
  Serial.begin(115200);
  // The ADC input range (or gain) can be changed via the following
  // functions, but be careful never to exceed VDD +0.3V max, or to
  // exceed the upper and lower limits if you adjust the input range!
  // Setting these values incorrectly may destroy your ADC!
  //                                                                ADS1015  ADS1115
  //                                                                -------  -------
  // ads.setGain(GAIN_TWOTHIRDS);  // 2/3x gain +/- 6.144V  1 bit = 3mV      0.1875mV (default)
  // ads.setGain(GAIN_ONE);        // 1x gain   +/- 4.096V  1 bit = 2mV      0.125mV
  // ads.setGain(GAIN_TWO);        // 2x gain   +/- 2.048V  1 bit = 1mV      0.0625mV
  // ads.setGain(GAIN_FOUR);       // 4x gain   +/- 1.024V  1 bit = 0.5mV    0.03125mV
  // ads.setGain(GAIN_EIGHT);      // 8x gain   +/- 0.512V  1 bit = 0.25mV   0.015625mV
  // ads.setGain(GAIN_SIXTEEN);    // 16x gain  +/- 0.256V  1 bit = 0.125mV  0.0078125mV
  Serial.println("Initializing ADS.");
  if (!ads.begin()) {
    Serial.println("Failed to initialize ADS.");
    while (1);
  }
}

void send_analog(uint8_t id, int16_t val) {
   uint8_t high = ((val >> 12) & 0x0F) |  (id << 4);
   uint8_t mid = (val >> 6) & 0x3F;
   uint8_t low  = val & 0x3F;
   Serial.write(high);
   Serial.write(mid);
   Serial.write(low);
   Serial.write(0xFF);  // Separator
}


void loop(void)
{
  int16_t pressure, force;
  unsigned long currentMicros = micros();
  if (currentMicros - previousMicros >= SAMPLING_INTERVAL_US) {
    previousMicros += SAMPLING_INTERVAL_US;
    pressure = ads.readADC_Differential_0_1();
    force = ads.readADC_SingleEnded(2);
    //Serial.print(pressure);
    //Serial.print("\t");
    //Serial.println(force);
    send_analog(0,pressure);
    send_analog(1,force);
  } 
}
