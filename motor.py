import RPi.GPIO as GPIO
from time import sleep

# ---------------------------
# GPIO 기본 설정
# ---------------------------
GPIO.setmode(GPIO.BCM)

# ---------------------------
# DC Motor Pin Mapping
# ---------------------------
ENA = 26   # Motor A Enable
IN1 = 19
IN2 = 13

ENB = 0    # Motor B Enable
IN3 = 6
IN4 = 5

# ---------------------------
# SERVO CONSTANTS
# ---------------------------
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3


# ==========================================================
# DC MOTOR FUNCTIONS  (Always ON / OFF version)
# ==========================================================

def init_dc_motors():
    """Initialize DC motor pins (no PWM)."""
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)

    GPIO.setup(ENB, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)

def dc_motor_on():
    """Turn DC motors ON at full power (no PWM)."""
    # Motor A
    GPIO.output(ENA, GPIO.HIGH)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

    # Motor B
    GPIO.output(ENB, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def dc_motor_off():
    """Turn DC motors OFF completely."""
    # Motor A
    GPIO.output(ENA, GPIO.LOW)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)

    # Motor B
    GPIO.output(ENB, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)


# ==========================================================
# SERVO FUNCTIONS
# ==========================================================

def init_servo(servo_pin):
    """Initialize a servo motor on given pin."""
    GPIO.setup(servo_pin, GPIO.OUT)
    servo = GPIO.PWM(servo_pin, 50)  # 50Hz
    servo.start(0)
    return servo

def setServoPos(servo, degree):
    """Move servo to an angle (0–180)."""
    if degree > 180:
        degree = 180
    if degree < 0:
        degree = 0

    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo.ChangeDutyCycle(duty)


# ==========================================================
# Test Code (optional)
# ==========================================================
if __name__ == "__main__":
    try:
        print("Initializing DC motors...")
        init_dc_motors()
        print("Turning motors ON...")
        dc_motor_on()
        sleep(3)

        print("Turning motors OFF...")
        dc_motor_off()

        print("Testing servo on pin 12...")
        servo = init_servo(12)
        setServoPos(servo, 0)
        sleep(1)
        setServoPos(servo, 90)
        sleep(1)
        setServoPos(servo, 180)
        sleep(1)
        servo.stop()

    except KeyboardInterrupt:
        pass

    finally:
        GPIO.cleanup()
        print("GPIO cleanup complete.")

