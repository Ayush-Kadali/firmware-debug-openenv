"""
Task definitions for the Firmware Debug Environment.

Each task simulates a real embedded firmware debugging scenario with:
- A simulated register map and peripheral state
- System logs with realistic (not hand-holding) information
- A known root cause and correct fix
- A grader that scores based on actual actions taken (not keywords)

Tasks progress from easy to hard:
  1. uart_baud_mismatch       — Easy   (single register misconfiguration)
  2. i2c_sensor_failure       — Medium (multi-factor peripheral failure)
  3. rtos_priority_inversion  — Hard   (concurrency bug in RTOS scheduling)
  4. dma_cache_coherency      — Hard   (DMA + cache interaction corruption)
  5. watchdog_reset_loop      — Medium (boot loop from watchdog misconfiguration)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Simulated Hardware Primitives
# ---------------------------------------------------------------------------

@dataclass
class Register:
    """A single hardware register."""
    name: str
    address: int
    value: int
    description: str
    writable: bool = True


@dataclass
class Peripheral:
    """A simulated MCU peripheral with registers."""
    name: str
    base_address: int
    description: str
    registers: Dict[str, Register] = field(default_factory=dict)
    status: str = "enabled"


@dataclass
class RTOSTask:
    """A simulated RTOS task."""
    name: str
    priority: int
    state: str  # running, ready, blocked, suspended
    stack_usage: int
    cpu_percent: float
    held_mutexes: List[str] = field(default_factory=list)
    waiting_on: Optional[str] = None
    description: str = ""


@dataclass
class TaskScenario:
    """Complete definition of a debugging task/scenario."""
    name: str
    difficulty: str
    description: str
    symptom: str
    peripherals: Dict[str, Peripheral]
    logs: List[str]
    rtos_tasks: Dict[str, RTOSTask]
    root_cause: str
    root_cause_keywords: List[str]  # for fallback text matching
    correct_fix: Dict[str, Any]
    key_registers: Set[str]  # registers the agent SHOULD read
    key_diagnostics: Set[str]  # diagnostic checks that help
    max_steps: int
    # Structural grading: specific register writes that constitute a correct fix
    required_writes: List[Dict[str, Any]] = field(default_factory=list)
    # Minimum writes needed for partial credit (e.g. 2 of 3 for I2C)
    min_writes_for_fix: int = 1
    # Dynamic state: registers that change after wrong writes
    write_side_effects: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task 1 — UART Baud Rate Mismatch (Easy)
# ---------------------------------------------------------------------------

def create_uart_baud_task() -> TaskScenario:
    """UART transmitting garbage data due to wrong baud rate register."""

    usart1 = Peripheral(
        name="USART1",
        base_address=0x40011000,
        description="UART serial port 1 — sends/receives data to external devices at a configured baud rate (speed)",
        registers={
            "CR1": Register("CR1", 0x40011000, 0x0000200C,
                           "Control register 1 — UE=1 (UART enabled), TE=1 (transmitter on), "
                           "RE=1 (receiver on). This UART is active and configured for TX+RX."),
            "CR2": Register("CR2", 0x40011004, 0x00000000,
                           "Control register 2 — STOP=00 means 1 stop bit (standard). "
                           "This is normal, unlikely to cause issues."),
            "CR3": Register("CR3", 0x40011008, 0x00000000,
                           "Control register 3 — no hardware flow control, no DMA. "
                           "Standard polling-mode UART operation."),
            "BRR": Register("BRR", 0x4001100C, 0x00000341,
                           "Baud Rate Register — controls communication speed. "
                           "Formula: baud_rate = peripheral_clock / BRR_value. "
                           "Current value 0x341 = 833 decimal. "
                           "If peripheral clock is 8 MHz: baud = 8000000 / 833 = 9604 (~9600 baud). "
                           "IMPORTANT: Both sides must use the same baud rate or data will be corrupted."),
            "ISR": Register("ISR", 0x40011010, 0x00C000D0,
                           "Status register (read-only). Key flags: "
                           "TXE=1 (TX buffer empty, ready to send), "
                           "TC=1 (transmission complete), "
                           "ORE=1 (OVERRUN ERROR — data arrived before previous was read, "
                           "often caused by baud rate mismatch between sender and receiver)",
                           writable=False),
            "RDR": Register("RDR", 0x40011014, 0x000000FF,
                           "Receive data register — last byte received: 0xFF (looks like garbage/noise)",
                           writable=False),
            "TDR": Register("TDR", 0x40011018, 0x00000041,
                           "Transmit data register — last byte sent: 0x41 = 'A' in ASCII"),
        },
    )

    rcc = Peripheral(
        name="RCC",
        base_address=0x40021000,
        description="Reset and Clock Control — manages all clock frequencies in the chip",
        registers={
            "CR": Register("CR", 0x40021000, 0x03035183,
                          "Clock control — HSI (internal 8 MHz oscillator) is ON and ready. "
                          "This is the main clock source."),
            "CFGR": Register("CFGR", 0x40021004, 0x00000000,
                            "Clock configuration — system clock = HSI (8 MHz), "
                            "AHB prescaler = 1 (no division), APB2 prescaler = 1 (no division). "
                            "Therefore: USART1 peripheral clock = 8 MHz."),
            "APB2ENR": Register("APB2ENR", 0x40021018, 0x00004001,
                               "APB2 peripheral clock enable — USART1 clock is ENABLED. "
                               "The UART has its clock and should be functional."),
        },
    )

    gpio_a = Peripheral(
        name="GPIOA",
        base_address=0x48000000,
        description="GPIO Port A — physical pin configuration for UART TX/RX lines",
        registers={
            "MODER": Register("MODER", 0x48000000, 0xEBFFFFFF,
                             "Pin mode register — PA9 and PA10 are set to 'alternate function' mode, "
                             "meaning they are connected to USART1 (TX and RX). This is correct."),
            "AFR_H": Register("AFR_H", 0x48000024, 0x00000770,
                             "Alternate function selection — PA9=AF7 (USART1_TX), PA10=AF7 (USART1_RX). "
                             "Pin routing is correct for UART operation."),
        },
    )

    # Logs: realistic but NOT hand-holding. Agent must deduce.
    logs = [
        "[0.000s] BOOT: STM32F4 firmware v2.1.3",
        "[0.001s] RCC: System clock source = HSI (8 MHz)",
        "[0.002s] RCC: AHB prescaler = 1, APB2 prescaler = 1",
        "[0.003s] GPIO: PA9 -> AF7, PA10 -> AF7",
        "[0.005s] USART1: Peripheral enabled, BRR=0x341",
        "[0.010s] MAIN: Starting sensor polling loop",
        "[0.100s] USART1: TX complete (12 bytes)",
        "[0.150s] USART1: RX timeout — no valid response",
        "[0.200s] USART1: TX complete (12 bytes)",
        "[0.250s] USART1: RX 12 bytes — CRC FAIL",
        "[0.300s] USART1: ISR=0x00C000D0 (ORE set)",
        "[0.301s] MAIN: Sensor comm error — 3/10 packets corrupted",
        "[0.500s] MAIN: External device confirmed operational at 19200 baud via separate test equipment",
        "[0.501s] MAIN: Expected communication: 19200 baud, 8N1 (8 data bits, no parity, 1 stop bit)",
    ]

    return TaskScenario(
        name="uart_baud_mismatch",
        difficulty="easy",
        description=(
            "The STM32 communicates with an external sensor module over USART1 at 19200 baud. "
            "Received data is corrupted — CRC failures, garbage bytes, and overrun errors. "
            "The external sensor module has been verified working correctly at 19200 baud with "
            "separate test equipment. Diagnose and fix the MCU-side configuration."
        ),
        symptom="UART receiving corrupted data — CRC failures and overrun errors",
        peripherals={"USART1": usart1, "RCC": rcc, "GPIOA": gpio_a},
        logs=logs,
        rtos_tasks={},
        root_cause="USART1 BRR register is 0x341 (9600 baud) but peripheral expects 19200 baud",
        root_cause_keywords=["baud", "brr", "9600", "19200", "0x341", "0x1a1", "mismatch"],
        correct_fix={
            "type": "set_register",
            "peripheral": "USART1",
            "register": "BRR",
            "value": 0x1A1,
        },
        key_registers={"USART1.BRR", "USART1.CR1", "USART1.ISR", "RCC.CFGR"},
        key_diagnostics={"check_connection"},
        max_steps=20,
        required_writes=[
            {"peripheral": "USART1", "register": "BRR", "value": 0x1A1, "alt_value": 417},
        ],
        min_writes_for_fix=1,
        write_side_effects={
            # Writing wrong BRR value causes more errors in log
            "USART1.BRR.wrong": {
                "new_logs": [
                    "[LIVE] USART1: BRR updated — retesting communication...",
                    "[LIVE] USART1: RX data still corrupted. Baud rate still incorrect.",
                ],
                "isr_update": 0x00C000D0,  # ORE stays set
            },
            "USART1.BRR.correct": {
                "new_logs": [
                    "[LIVE] USART1: BRR updated to 0x1A1",
                    "[LIVE] USART1: Communication restored — CRC check PASS",
                    "[LIVE] MAIN: Sensor data valid. System operational.",
                ],
                "isr_update": 0x000000C0,  # ORE cleared
            },
        },
    )


# ---------------------------------------------------------------------------
# Task 2 — I2C Sensor Failure (Medium)
# ---------------------------------------------------------------------------

def create_i2c_sensor_task() -> TaskScenario:
    """I2C sensor not responding — multiple contributing factors."""

    i2c1 = Peripheral(
        name="I2C1",
        base_address=0x40005400,
        description="I2C bus interface 1 — a two-wire protocol (SCL clock + SDA data) used to communicate with sensors. "
                    "The master (MCU) sends a 7-bit address to select which sensor to talk to.",
        registers={
            "CR1": Register("CR1", 0x40005400, 0x00000001,
                           "Control register 1 — PE=1 (peripheral enabled). "
                           "I2C interface is turned on and active."),
            "CR2": Register("CR2", 0x40005404, 0x04002068,
                           "Control register 2 — contains the target slave address and transfer config. "
                           "SADD field (bits 9:0) = 0x68 = the 7-bit address the MCU sends on the bus. "
                           "NBYTES=2 (requesting 2 bytes), RD_WRN=0 (write mode for address phase). "
                           "NOTE: I2C devices have a fixed base address, but the lowest bit may be "
                           "set by a hardware pin (often called AD0 or A0). Check schematic for pin wiring."),
            "TIMINGR": Register("TIMINGR", 0x40005408, 0x00100002,
                               "Timing register — controls the I2C clock (SCL) frequency. "
                               "Current config: PRESC=0, SCLL=0x02, SCLH=0x01. "
                               "With 8 MHz input clock, this produces SCL ≈ 1.33 MHz. "
                               "WARNING: I2C standard mode max is 100 kHz, fast mode max is 400 kHz. "
                               "1.33 MHz FAR EXCEEDS the maximum spec. Most sensors will not respond."),
            "ISR": Register("ISR", 0x40005418, 0x00000009,
                           "Status register (read-only). "
                           "NACKF=1 (NACK RECEIVED — the addressed device did NOT acknowledge). "
                           "This means either: wrong address, device not present, or bus error. "
                           "BUSY=0 (bus is idle now after failed attempt).",
                           writable=False),
            "OAR1": Register("OAR1", 0x4000540C, 0x00000000,
                            "Own address register — only used in I2C slave mode, not relevant here."),
        },
    )

    gpio_b = Peripheral(
        name="GPIOB",
        base_address=0x48000400,
        description="GPIO Port B — physical pin configuration for I2C1 lines (PB6=SCL clock, PB7=SDA data)",
        registers={
            "MODER": Register("MODER", 0x48000400, 0xFFFFA_FFF,
                             "Pin mode register — PB6 and PB7 are in 'alternate function' mode, "
                             "correctly routed to I2C1 peripheral."),
            "OTYPER": Register("OTYPER", 0x48000404, 0x00000000,
                              "Output type register — controls whether pins drive push-pull or open-drain. "
                              "Bit 6 (PB6/SCL) = 0 = push-pull. Bit 7 (PB7/SDA) = 0 = push-pull. "
                              "CRITICAL: I2C REQUIRES open-drain outputs (bits should be 1, not 0). "
                              "Push-pull mode can cause bus contention and signal corruption. "
                              "Correct value should have bits 6 and 7 set: 0xC0."),
            "PUPDR": Register("PUPDR", 0x48000408, 0x00000000,
                             "Pull-up/pull-down register — 00 for PB6,PB7 means no internal pull-ups. "
                             "External pull-up resistors on the board may compensate."),
            "AFR_L": Register("AFR_L", 0x48000420, 0x44000000,
                             "Alternate function — PB6=AF4 (I2C1_SCL), PB7=AF4 (I2C1_SDA). "
                             "Pin routing is correct."),
        },
    )

    rcc = Peripheral(
        name="RCC",
        base_address=0x40021000,
        description="Reset and Clock Control — manages chip clocking",
        registers={
            "APB1ENR": Register("APB1ENR", 0x40021040, 0x00200000,
                               "APB1 clock enable — I2C1 clock is ENABLED."),
            "CFGR": Register("CFGR", 0x40021004, 0x00000000,
                            "Clock config — system clock = HSI (8 MHz), no prescaler. "
                            "I2C1 peripheral clock = 8 MHz."),
        },
    )

    # Logs: informational but require deduction
    logs = [
        "[0.000s] BOOT: STM32F4 firmware v2.1.3",
        "[0.005s] I2C1: Peripheral initialized",
        "[0.006s] I2C1: Configured for address 0x68",
        "[0.010s] I2C1: START condition sent",
        "[0.011s] I2C1: NACK received on address byte",
        "[0.012s] I2C1: Retry 2/5 — NACK",
        "[0.013s] I2C1: Retry 3/5 — NACK",
        "[0.014s] I2C1: Retry 4/5 — NACK",
        "[0.015s] I2C1: Retry 5/5 — NACK",
        "[0.020s] I2C1: Slave device not responding",
        "[0.050s] I2C1: Bus recovery — 9 clock pulses",
        "[0.051s] I2C1: Still NACK on address phase",
        "[0.100s] HW: Schematic note — U3 (IMU) AD0 pin connected to VCC via 10k",
        "[0.101s] HW: I2C1 routed to PB6(SCL), PB7(SDA)",
    ]

    return TaskScenario(
        name="i2c_sensor_failure",
        difficulty="medium",
        description=(
            "An IMU sensor connected over I2C1 is not responding. The MCU receives NACKs "
            "during the address phase on every attempt. The sensor is confirmed powered "
            "(3.3V supply verified). The board has external 4.7k pull-up resistors on SDA/SCL. "
            "There may be multiple issues. Identify and fix all of them."
        ),
        symptom="I2C sensor not responding — NACK on every address attempt",
        peripherals={"I2C1": i2c1, "GPIOB": gpio_b, "RCC": rcc},
        logs=logs,
        rtos_tasks={},
        root_cause=(
            "Three issues: (1) Wrong slave address (0x68 but AD0=VCC means 0x69), "
            "(2) I2C clock too fast (TIMINGR gives ~1.3MHz, max 400kHz), "
            "(3) GPIO output type is push-pull instead of open-drain"
        ),
        root_cause_keywords=[
            "address", "0x69", "0x68", "ad0", "timing", "clock", "fast", "400",
            "open-drain", "push-pull", "otyper",
        ],
        correct_fix={
            "type": "multi_fix",
            "fixes": [
                {"peripheral": "I2C1", "register": "CR2", "field": "SADD", "value": 0x69},
                {"peripheral": "I2C1", "register": "TIMINGR", "value": 0x00310309},
                {"peripheral": "GPIOB", "register": "OTYPER", "value": 0x000000C0},
            ],
        },
        key_registers={
            "I2C1.CR2", "I2C1.TIMINGR", "I2C1.ISR",
            "GPIOB.OTYPER", "GPIOB.MODER", "GPIOB.PUPDR",
        },
        key_diagnostics={"check_connection", "run_diagnostic"},
        max_steps=25,
        required_writes=[
            {"peripheral": "I2C1", "register": "CR2", "check": "sadd_0x69"},
            {"peripheral": "I2C1", "register": "TIMINGR", "check": "timing_valid"},
            {"peripheral": "GPIOB", "register": "OTYPER", "check": "open_drain"},
        ],
        min_writes_for_fix=2,  # Must fix at least 2 of 3 to pass
        write_side_effects={
            "I2C1.CR2.any": {
                "new_logs": ["[LIVE] I2C1: CR2 updated — address configuration changed"],
            },
            "I2C1.TIMINGR.any": {
                "new_logs": ["[LIVE] I2C1: TIMINGR updated — clock reconfigured"],
            },
            "GPIOB.OTYPER.any": {
                "new_logs": ["[LIVE] GPIO: OTYPER updated — output type changed"],
            },
        },
    )


# ---------------------------------------------------------------------------
# Task 3 — RTOS Priority Inversion (Hard)
# ---------------------------------------------------------------------------

def create_rtos_priority_task() -> TaskScenario:
    """Priority inversion causing intermittent sensor data corruption."""

    spi1 = Peripheral(
        name="SPI1",
        base_address=0x40013000,
        description="SPI bus 1 — high-speed serial interface shared between barometer sensor and flash memory",
        registers={
            "CR1": Register("CR1", 0x40013000, 0x0000034C,
                           "SPI control — enabled, master mode, clock divider /16. "
                           "The SPI hardware itself is configured correctly."),
            "SR": Register("SR", 0x40013008, 0x00000003,
                          "SPI status — TXE=1 (ready to send), RXNE=1 (data available). "
                          "No hardware errors. SPI peripheral is healthy.",
                          writable=False),
            "DR": Register("DR", 0x4001300C, 0x00005A23,
                          "Last data read via SPI — 0x5A23 (raw barometer pressure reading)"),
        },
    )

    adc1 = Peripheral(
        name="ADC1",
        base_address=0x40012000,
        description="Analog-to-Digital Converter — reads analog voltages (temperature, battery)",
        registers={
            "CR": Register("CR", 0x40012000, 0x20000001,
                          "ADC control — enabled, conversion started"),
            "ISR": Register("ISR", 0x40012004, 0x00000004,
                           "ADC status — conversion complete, reading available",
                           writable=False),
            "DR": Register("DR", 0x40012040, 0x00000A3C,
                          "ADC data — 2620 counts (temperature reading). ADC is working fine.",
                          writable=False),
        },
    )

    tasks = {
        "sensor_read": RTOSTask(
            name="sensor_read",
            priority=3,
            state="blocked",
            stack_usage=512,
            cpu_percent=15.2,
            held_mutexes=[],
            waiting_on="spi_mutex",
            description=(
                "HIGHEST priority task (pri=3). Reads barometer sensor via SPI1 every 100ms. "
                "Currently BLOCKED — waiting to acquire spi_mutex. "
                "Cannot run until spi_mutex is released by whoever holds it."
            ),
        ),
        "telemetry_tx": RTOSTask(
            name="telemetry_tx",
            priority=2,
            state="running",
            stack_usage=1024,
            cpu_percent=45.8,
            held_mutexes=[],
            waiting_on=None,
            description=(
                "MEDIUM priority task (pri=2). Formats telemetry packets and sends via UART. "
                "Currently RUNNING — doing heavy string formatting (~50ms per cycle). "
                "Does NOT need the SPI bus or spi_mutex. "
                "NOTE: In a preemptive RTOS, higher-priority tasks should preempt lower ones, "
                "but this task (pri=2) is running while sensor_read (pri=3) is blocked."
            ),
        ),
        "data_logger": RTOSTask(
            name="data_logger",
            priority=1,
            state="ready",
            stack_usage=2048,
            cpu_percent=30.1,
            held_mutexes=["spi_mutex"],
            waiting_on=None,
            description=(
                "LOWEST active priority task (pri=1). Logs sensor data to flash memory via SPI1. "
                "State=READY (wants to run but can't — preempted by telemetry_tx). "
                "HOLDS spi_mutex — acquired it before being preempted. "
                "Cannot release spi_mutex until it finishes its flash write, "
                "but cannot run because telemetry_tx (pri=2) has preempted it."
            ),
        ),
        "idle": RTOSTask(
            name="idle",
            priority=0,
            state="ready",
            stack_usage=128,
            cpu_percent=8.9,
            description="System idle task (pri=0). Runs when nothing else needs CPU.",
        ),
    }

    mutex_info = Peripheral(
        name="RTOS_MUTEXES",
        base_address=0x00000000,
        description="RTOS mutex/lock state — mutexes protect shared resources from concurrent access",
        registers={
            "spi_mutex": Register("spi_mutex", 0x0, 0x00000001,
                                 "SPI1 bus lock — LOCKED. "
                                 "Holder: data_logger (priority 1, the LOWEST active task). "
                                 "Waiter: sensor_read (priority 3, the HIGHEST priority task). "
                                 "Type: binary semaphore — this type does NOT support priority inheritance. "
                                 "With priority inheritance, the holder's priority would be temporarily "
                                 "raised to match the highest waiter, preventing medium-priority tasks "
                                 "from preempting it. Currently that is NOT happening.",
                                 writable=False),
            "flash_mutex": Register("flash_mutex", 0x4, 0x00000000,
                                   "Flash write lock — UNLOCKED (not in use)", writable=False),
            "uart_mutex": Register("uart_mutex", 0x8, 0x00000000,
                                  "UART TX lock — UNLOCKED (not in use)", writable=False),
        },
    )

    rtos_info = Peripheral(
        name="RTOS_CONFIG",
        base_address=0x00000000,
        description="RTOS kernel configuration — how the operating system schedules tasks",
        registers={
            "SCHEDULER": Register("SCHEDULER", 0x0, 0x00000001,
                                 "Scheduler: RUNNING in preemptive mode (higher priority always runs first), "
                                 "tick rate=1000Hz, time slicing=ON."),
            "HEAP_FREE": Register("HEAP_FREE", 0x4, 0x00003A00,
                                 "Free heap memory: 14848 / 32768 bytes — sufficient, no memory pressure."),
            "TASK_COUNT": Register("TASK_COUNT", 0x8, 0x00000004,
                                  "4 active tasks in the system."),
            "MUTEX_PROTOCOL": Register("MUTEX_PROTOCOL", 0xC, 0x00000000,
                                      "Mutex priority protocol: NONE (value 0). "
                                      "Options: 0=none, 1=priority inheritance, 2=priority ceiling. "
                                      "Priority inheritance means: when a high-priority task waits on a mutex, "
                                      "the holder's priority is temporarily boosted to prevent being preempted "
                                      "by medium-priority tasks. Currently DISABLED because the mutex was created "
                                      "as a binary semaphore instead of a proper mutex."),
        },
    )

    # Logs: realistic RTOS debug output — no "priority inversion" mentioned
    logs = [
        "[0.000s] BOOT: STM32H7 firmware v3.0.1 — FreeRTOS 10.4.3",
        "[0.001s] RTOS: Task created: sensor_read (pri=3, stack=512)",
        "[0.001s] RTOS: Task created: telemetry_tx (pri=2, stack=1024)",
        "[0.002s] RTOS: Task created: data_logger (pri=1, stack=2048)",
        "[0.002s] RTOS: spi_mutex created (binary semaphore)",
        "[0.003s] RTOS: Scheduler started",
        "[0.100s] sensor_read: Barometer OK — 101325 Pa",
        "[0.200s] sensor_read: Barometer OK — 101324 Pa",
        "[0.300s] sensor_read: Barometer OK — 101326 Pa",
        "[0.400s] data_logger: Acquired spi_mutex — flash write starting (4KB)",
        "[0.410s] telemetry_tx: Woke — formatting packet",
        "[0.415s] RTOS: Context switch: data_logger -> telemetry_tx",
        "[0.420s] sensor_read: Attempting spi_mutex acquire...",
        "[0.421s] sensor_read: BLOCKED — spi_mutex unavailable",
        "[0.460s] telemetry_tx: Still formatting (heavy string ops)",
        "[0.500s] sensor_read: WARNING — missed 100ms deadline (blocked 80ms)",
        "[0.510s] telemetry_tx: TX complete",
        "[0.511s] RTOS: Context switch: telemetry_tx -> data_logger",
        "[0.550s] data_logger: Flash write complete — released spi_mutex",
        "[0.551s] sensor_read: Acquired spi_mutex",
        "[0.552s] sensor_read: WARNING — stale data, 150ms since last read",
        "[0.553s] sensor_read: Pressure=101400 Pa (jump of 74 Pa)",
        "[1.000s] MAIN: sensor_read missed 6/10 deadlines",
        "[1.001s] MAIN: Pressure spikes correlate with flash write activity",
    ]

    return TaskScenario(
        name="rtos_priority_inversion",
        difficulty="hard",
        description=(
            "A multi-task FreeRTOS system has three tasks sharing an SPI bus via a mutex: "
            "a high-priority sensor reader, a medium-priority telemetry formatter, and a "
            "low-priority data logger. The sensor reader is intermittently missing its "
            "100ms deadline, resulting in stale readings and data jumps. The issue only "
            "manifests when the data logger is actively writing to flash. Identify the "
            "concurrency bug and apply the correct fix."
        ),
        symptom="High-priority task intermittently missing deadlines, correlated with flash writes",
        peripherals={
            "SPI1": spi1, "ADC1": adc1,
            "RTOS_MUTEXES": mutex_info, "RTOS_CONFIG": rtos_info,
        },
        logs=logs,
        rtos_tasks=tasks,
        root_cause=(
            "Priority inversion: data_logger(pri=1) holds spi_mutex, telemetry_tx(pri=2) "
            "preempts it, so sensor_read(pri=3) is blocked while a lower-priority task runs. "
            "Fix: use a proper mutex (xSemaphoreCreateMutex) instead of binary semaphore "
            "to enable priority inheritance."
        ),
        root_cause_keywords=[
            "priority inversion", "inheritance", "inversion",
            "mutex", "preempt", "semaphore", "binary", "xSemaphoreCreateMutex",
        ],
        correct_fix={
            "type": "enable_priority_inheritance",
            "target": "spi_mutex",
        },
        key_registers={
            "RTOS_MUTEXES.spi_mutex", "RTOS_CONFIG.MUTEX_PROTOCOL",
            "RTOS_CONFIG.SCHEDULER",
        },
        key_diagnostics={"analyze_task", "run_diagnostic"},
        max_steps=30,
        required_writes=[],  # This task uses apply_fix, not register writes
        min_writes_for_fix=0,
    )


# ---------------------------------------------------------------------------
# Task 4 — DMA + Cache Coherency (Hard)
# ---------------------------------------------------------------------------

def create_dma_cache_task() -> TaskScenario:
    """DMA transfer data corruption due to cache coherency issue on Cortex-M7."""

    dma1 = Peripheral(
        name="DMA1",
        base_address=0x40026000,
        description="DMA (Direct Memory Access) — transfers data between peripherals and memory "
                    "WITHOUT involving the CPU. The CPU is free to do other work while DMA moves data. "
                    "IMPORTANT: DMA writes directly to physical memory, bypassing any CPU caches.",
        registers={
            "S0CR": Register("S0CR", 0x40026010, 0x0E010C19,
                            "DMA stream 0 config — enabled, transfer-complete interrupt on, "
                            "direction = peripheral-to-memory, 32-bit transfers, very high priority. "
                            "DMA hardware configuration looks correct."),
            "S0NDTR": Register("S0NDTR", 0x40026014, 0x00000100,
                              "Transfer count: 256 items per DMA transfer."),
            "S0PAR": Register("S0PAR", 0x40026018, 0x4001300C,
                             "Source: SPI1 data register (the ADC sends samples via SPI, "
                             "DMA reads them automatically)."),
            "S0M0AR": Register("S0M0AR", 0x4002601C, 0x20010000,
                              "Destination: memory address 0x20010000. "
                              "This is where DMA writes the ADC samples. "
                              "The CPU later reads from this same address to process the data."),
            "LISR": Register("LISR", 0x40026000, 0x00000020,
                            "DMA status — TCIF0=1: transfer complete for stream 0. "
                            "The DMA HAS successfully written all 256 samples to memory.",
                            writable=False),
        },
    )

    spi1 = Peripheral(
        name="SPI1",
        base_address=0x40013000,
        description="SPI bus 1 — receives ADC samples, DMA-enabled",
        registers={
            "CR1": Register("CR1", 0x40013000, 0x0000034C,
                           "SPI control — enabled, master mode. SPI is working fine."),
            "CR2": Register("CR2", 0x40013004, 0x00001700,
                           "SPI control 2 — RXDMAEN=1 (receive DMA enabled). "
                           "SPI automatically feeds received data to DMA for memory transfer."),
            "SR": Register("SR", 0x40013008, 0x00000002,
                          "SPI status — healthy, no errors.", writable=False),
        },
    )

    scb = Peripheral(
        name="SCB",
        base_address=0xE000ED00,
        description="System Control Block — CPU core configuration including CACHES. "
                    "This Cortex-M7 has a 32KB data cache (D-cache) between CPU and memory. "
                    "When the CPU reads an address, it first checks the cache. If the data is "
                    "in cache, it uses the cached copy (fast). If not, it fetches from memory (slow). "
                    "PROBLEM: If something ELSE (like DMA) writes to memory, the cache still holds the OLD data.",
        registers={
            "CCR": Register("CCR", 0xE000ED14, 0x00040200,
                           "CPU config — D-cache: ENABLED, I-cache: ENABLED. "
                           "The data cache is ON, meaning the CPU may read stale cached values "
                           "instead of fresh data written by DMA."),
            "DCISW": Register("DCISW", 0xE000EF60, 0x00000000,
                             "D-cache INVALIDATE register (write-only). "
                             "Writing to this register forces the CPU to discard cached data "
                             "and re-fetch from actual memory on next read. "
                             "This is how you tell the CPU: 'your cache is outdated, re-read from memory.'"),
            "DCCISW": Register("DCCISW", 0xE000EF74, 0x00000000,
                              "D-cache clean + invalidate register (write-only). "
                              "Writes dirty cache lines back to memory AND invalidates them."),
        },
    )

    mpu = Peripheral(
        name="MPU",
        base_address=0xE000ED90,
        description="Memory Protection Unit — defines caching policy per memory region. "
                    "Each region can be: non-cacheable (DMA-safe), write-through, or write-back (fastest but risky with DMA).",
        registers={
            "CTRL": Register("CTRL", 0xE000ED94, 0x00000005,
                            "MPU enabled with default background map."),
            "RNR": Register("RNR", 0xE000ED98, 0x00000000,
                           "Selected MPU region: 0 (the DMA buffer region)."),
            "RBAR": Register("RBAR", 0xE000ED9C, 0x20010000,
                            "Region 0 base address: 0x20010000 — this is exactly where DMA writes data."),
            "RASR": Register("RASR", 0xE000EDA0, 0x0308003F,
                            "Region 0 attributes: size=256KB, C=1 (cacheable), B=1 (bufferable). "
                            "This means WRITE-BACK cache policy. "
                            "With write-back: CPU reads come from cache, NOT from actual memory. "
                            "If DMA updates memory behind the cache's back, the CPU sees STALE data. "
                            "Fix options: (1) invalidate D-cache after each DMA transfer, or "
                            "(2) change this region to non-cacheable (set C=0, B=0)."),
        },
    )

    logs = [
        "[0.000s] BOOT: STM32H743 firmware v1.2.0 — Cortex-M7 @ 480MHz",
        "[0.001s] CACHE: D-cache enabled (32KB, 4-way set associative)",
        "[0.002s] CACHE: I-cache enabled",
        "[0.003s] MPU: Region 0 configured — 0x20010000, 256KB, cacheable/bufferable",
        "[0.004s] DMA1: Stream 0 configured for SPI1_RX -> memory",
        "[0.005s] SPI1: Peripheral enabled, RX DMA enabled",
        "[0.010s] MAIN: Starting ADC acquisition loop via DMA",
        "[0.100s] DMA1: Transfer complete — 256 samples to 0x20010000",
        "[0.101s] MAIN: Processing buffer — first 4 samples: 0x0A3C, 0x0A3D, 0x0A3B, 0x0A3C",
        "[0.200s] DMA1: Transfer complete — 256 samples to 0x20010000",
        "[0.201s] MAIN: Processing buffer — first 4 samples: 0x0A3C, 0x0A3C, 0x0A3C, 0x0A3C",
        "[0.202s] MAIN: WARNING — all values identical (stale data?)",
        "[0.300s] DMA1: Transfer complete — 256 samples",
        "[0.301s] MAIN: First 4 samples: 0x0A3C, 0x0A3C, 0x0A3C, 0x0A3C",
        "[0.302s] MAIN: ERROR — data not updating despite DMA TCIF=1",
        "[0.400s] DEBUG: Reading DMA buffer directly with debugger shows correct varying data",
        "[0.401s] DEBUG: But CPU reads same address and gets stale 0x0A3C repeatedly",
    ]

    return TaskScenario(
        name="dma_cache_coherency",
        difficulty="hard",
        description=(
            "An STM32H7 (Cortex-M7 @ 480MHz) acquires ADC samples via DMA into a memory "
            "buffer at 0x20010000. The DMA transfer complete flag fires correctly, and a "
            "debugger memory read confirms fresh data in the buffer. However, the CPU "
            "consistently reads stale values from the same address. The system was working "
            "before D-cache was enabled. Identify the root cause and fix it."
        ),
        symptom="CPU reads stale data from DMA buffer despite DMA transferring correct data",
        peripherals={"DMA1": dma1, "SPI1": spi1, "SCB": scb, "MPU": mpu},
        logs=logs,
        rtos_tasks={},
        root_cause=(
            "D-cache coherency issue: DMA writes to memory bypass the CPU cache. The MPU "
            "region for the DMA buffer is configured as cacheable/bufferable (write-back), "
            "so the CPU reads from cache (stale) instead of memory (fresh DMA data). Fix: "
            "either invalidate D-cache after DMA transfer, or configure the MPU region as "
            "non-cacheable for the DMA buffer."
        ),
        root_cause_keywords=[
            "cache", "coherency", "coherence", "d-cache", "dcache",
            "invalidate", "non-cacheable", "write-back", "write-through",
            "mpu", "dma", "stale", "bypass",
        ],
        correct_fix={
            "type": "cache_fix",
            "options": [
                "invalidate_dcache",  # SCB_InvalidateDCache_by_Addr
                "mpu_non_cacheable",  # Set MPU region to non-cacheable
            ],
        },
        key_registers={
            "SCB.CCR", "MPU.RASR", "MPU.RBAR",
            "DMA1.S0M0AR", "DMA1.S0CR",
        },
        key_diagnostics={"run_diagnostic"},
        max_steps=25,
        required_writes=[
            # Option A: invalidate cache
            {"peripheral": "SCB", "register": "DCISW", "check": "cache_invalidate"},
            # Option B: change MPU to non-cacheable
            {"peripheral": "MPU", "register": "RASR", "check": "non_cacheable"},
        ],
        min_writes_for_fix=1,  # Either option is sufficient
    )


# ---------------------------------------------------------------------------
# Task 5 — Watchdog Reset Loop (Medium)
# ---------------------------------------------------------------------------

def create_watchdog_task() -> TaskScenario:
    """System stuck in boot loop due to watchdog reset from initialization ordering."""

    iwdg = Peripheral(
        name="IWDG",
        base_address=0x40003000,
        description="Independent Watchdog Timer — a safety mechanism that resets the chip if software hangs. "
                    "The software must periodically 'feed' (reload) the watchdog before it counts down to zero. "
                    "If the watchdog reaches zero, it forces a full system reset. Once started, it CANNOT be stopped.",
        registers={
            "KR": Register("KR", 0x40003000, 0x0000CCCC,
                          "Key register (write-only commands): "
                          "Write 0xCCCC = START the watchdog (already done — watchdog is running!). "
                          "Write 0xAAAA = FEED/RELOAD the watchdog (reset the countdown). "
                          "Write 0x5555 = UNLOCK PR and RLR registers for modification."),
            "PR": Register("PR", 0x40003004, 0x00000001,
                          "Prescaler register — value 1 means clock divider = /8. "
                          "Watchdog clock = LSI (40 kHz) / 8 = 5000 Hz. "
                          "Formula: divider = 4 * 2^(PR+2) = 4 * 2^3 = 32."),
            "RLR": Register("RLR", 0x40003008, 0x00000064,
                           "Reload register — countdown start value = 100 (0x64). "
                           "Watchdog timeout = divider * RLR / LSI_freq = 32 * 100 / 40000 = 0.08 seconds = 80ms. "
                           "This means: if software doesn't feed the watchdog within 80ms, the system RESETS. "
                           "To increase timeout: write a larger value here (after unlocking with KR=0x5555)."),
            "SR": Register("SR", 0x4000300C, 0x00000000,
                          "Status — no pending register updates.",
                          writable=False),
        },
    )

    flash = Peripheral(
        name="FLASH",
        base_address=0x40022000,
        description="Flash memory interface — controls how the CPU reads program memory. "
                    "At high clock speeds, flash memory is slower than the CPU and needs 'wait states' "
                    "(extra clock cycles to wait for data). Wrong wait states = unreliable reads.",
        registers={
            "ACR": Register("ACR", 0x40022000, 0x00000000,
                           "Access control — LATENCY bits [2:0] = 0 (ZERO wait states). "
                           "At 72 MHz system clock, flash needs at LEAST 2 wait states. "
                           "0 wait states at 72 MHz causes unreliable flash reads, "
                           "which slows down code execution as the CPU retries failed reads."),
            "SR": Register("SR", 0x40022004, 0x00000000,
                          "Flash status — no errors reported (but timing violations may not flag here).",
                          writable=False),
        },
    )

    rcc = Peripheral(
        name="RCC",
        base_address=0x40021000,
        description="Reset and Clock Control — manages clock sources and tracks reset causes",
        registers={
            "CR": Register("CR", 0x40021000, 0x00000083,
                          "Clock control — HSI (8 MHz internal oscillator) is ON and ready. "
                          "PLL is configured but PLLRDY=0 (not yet locked). "
                          "PLL lock can take 50-60ms from cold start."),
            "CFGR": Register("CFGR", 0x40021004, 0x001D0402,
                            "Clock config — target: PLL at 72 MHz (HSE 8MHz * 9 = 72MHz). "
                            "System clock is set to PLL source. "
                            "APB1 = 36 MHz (/2), APB2 = 72 MHz (/1)."),
            "CSR": Register("CSR", 0x40021024, 0x24000000,
                           "Reset cause register (READ-ONLY, very important!). "
                           "IWDGRSTF = 1: the LAST RESET was caused by the WATCHDOG. "
                           "PORRSTF = 1: power-on reset also detected (normal at first boot). "
                           "This tells you the system was reset by the watchdog timer, "
                           "confirming the boot loop is caused by watchdog timeout.",
                           writable=False),
            "BDCR": Register("BDCR", 0x40021020, 0x00000000,
                            "Backup domain control — LSE oscillator off (not relevant)."),
        },
    )

    gpio_c = Peripheral(
        name="GPIOC",
        base_address=0x40011000,
        description="GPIO Port C — PC13 = LED",
        registers={
            "ODR": Register("ODR", 0x40011010, 0x00002000,
                           "Output data register — PC13=1 (LED off, active low)"),
            "CRH": Register("CRH", 0x40011004, 0x00300000,
                           "Configuration register high — PC13 = push-pull output 2MHz"),
        },
    )

    usart1 = Peripheral(
        name="USART1",
        base_address=0x40013800,
        description="USART1 — debug console",
        registers={
            "CR1": Register("CR1", 0x40013800, 0x00000000,
                           "Control register 1 — UE=0 (USART not yet enabled)"),
            "BRR": Register("BRR", 0x40013808, 0x00000000,
                           "Baud rate register — not configured yet"),
            "SR": Register("SR", 0x40013800, 0x00C00000,
                          "Status register", writable=False),
        },
    )

    # The story: IWDG starts before peripherals are initialized. PLL config
    # takes too long, watchdog fires during boot, system resets.
    logs = [
        "[0.000s] BOOT: STM32F103 firmware v1.0.4",
        "[0.001s] RCC: CSR = 0x24000000",
        "[0.002s] BOOT: Reset cause: IWDG (watchdog reset detected)",
        "[0.003s] BOOT: Reset counter: 47 (incrementing rapidly)",
        "[0.004s] IWDG: Watchdog started (KR=0xCCCC written)",
        "[0.005s] RCC: Switching to PLL...",
        "[0.006s] RCC: Waiting for PLL lock...",
        "[0.030s] RCC: PLL still not ready (HSE startup delay)",
        "[0.060s] RCC: PLL locked, SWS=10",
        "[0.061s] FLASH: Configuring wait states...",
        "[0.062s] FLASH: ACR = 0x00000000 (0 wait states)",
        "[0.063s] USART1: Initializing debug console...",
        "[0.080s] GPIO: Configuring LED on PC13",
        "[0.090s] MAIN: System initialization incomplete...",
        "[0.099s] IWDG: >>> WATCHDOG TIMEOUT — SYSTEM RESET <<<",
    ]

    return TaskScenario(
        name="watchdog_reset_loop",
        difficulty="medium",
        description=(
            "An STM32F103 is stuck in a boot loop, resetting approximately every 100ms. "
            "The reset counter shows 47 rapid resets. The system partially initializes "
            "but never reaches the main application loop. The hardware is a new board "
            "revision where the Independent Watchdog was added to the startup sequence. "
            "Determine why the system resets before completing initialization and fix it."
        ),
        symptom="System stuck in boot loop — resets every ~100ms, never reaches main loop",
        peripherals={
            "IWDG": iwdg, "FLASH": flash, "RCC": rcc,
            "GPIOC": gpio_c, "USART1": usart1,
        },
        logs=logs,
        rtos_tasks={},
        root_cause=(
            "Watchdog timeout is too short for the boot sequence. IWDG is started at "
            "0.004s but PLL lock takes until 0.060s. IWDG config: prescaler=4 (/64), "
            "reload=100. Timeout = (64 * 100) / 40000 = 0.16s. But total init including "
            "PLL wait + flash + USART takes ~95ms, and the watchdog isn't being fed during "
            "init. Additionally, flash wait states = 0 at 72MHz is wrong (needs 2), causing "
            "sporadic flash read errors that slow init further. "
            "Fix: either increase IWDG timeout (increase RLR), or feed watchdog during "
            "init, AND fix flash wait states."
        ),
        root_cause_keywords=[
            "watchdog", "iwdg", "timeout", "boot", "pll", "startup",
            "reload", "rlr", "prescaler", "feed", "kick",
            "flash", "wait state", "latency",
        ],
        correct_fix={
            "type": "multi_fix",
            "fixes": [
                {"peripheral": "IWDG", "register": "RLR", "description": "Increase reload value"},
                {"peripheral": "FLASH", "register": "ACR", "description": "Set correct wait states"},
            ],
        },
        key_registers={
            "IWDG.KR", "IWDG.PR", "IWDG.RLR",
            "FLASH.ACR", "RCC.CSR", "RCC.CR", "RCC.CFGR",
        },
        key_diagnostics={"run_diagnostic", "check_connection"},
        max_steps=25,
        required_writes=[
            {"peripheral": "IWDG", "register": "RLR", "check": "rlr_increase"},
            {"peripheral": "FLASH", "register": "ACR", "check": "wait_states"},
        ],
        min_writes_for_fix=1,  # Fixing either helps, both is ideal
    )


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "uart_baud_mismatch": create_uart_baud_task,
    "i2c_sensor_failure": create_i2c_sensor_task,
    "rtos_priority_inversion": create_rtos_priority_task,
    "dma_cache_coherency": create_dma_cache_task,
    "watchdog_reset_loop": create_watchdog_task,
}

TASK_LIST = list(TASK_REGISTRY.keys())
