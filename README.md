---
title: Firmware Debug Environment
emoji: 🔧
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Firmware Debug Environment

An OpenEnv environment where AI agents debug **real embedded firmware issues** on a simulated ARM Cortex-M microcontroller (STM32). Agents interact with hardware registers, system logs, peripheral diagnostics, and RTOS task states to diagnose and fix bugs — exactly like a firmware engineer would.

## Why This Environment?

Firmware debugging is one of the hardest real-world tasks in embedded systems engineering. It requires:
- Decoding hardware register bitfields and cross-referencing datasheets
- Correlating system logs with peripheral state
- Understanding clock trees, baud rate calculations, and protocol timing
- Diagnosing concurrency bugs in RTOS scheduling
- Reasoning about memory hierarchy (cache coherency, DMA)

**No toy problems.** Every task in this environment is a real bug that embedded engineers encounter in production — from misconfigured baud rates to priority inversion to DMA cache coherency issues.

## Environment Overview

The agent connects to a simulated STM32 MCU experiencing a fault. Through systematic debugging actions, the agent must **investigate**, **diagnose**, and **fix** the issue. The simulation is **dynamic** — register writes mutate system state, produce observable consequences, and can cause cascading failures if incorrect.

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `read_register` | `target`, `register` | Read a peripheral register (value + description) |
| `write_register` | `target`, `register`, `value` | Write a value — **mutates system state dynamically** |
| `list_peripherals` | — | List all peripherals and their register maps |
| `read_log` | — | View system boot/runtime logs (new entries appear after writes) |
| `check_connection` | `target` | Physical connection diagnostics (signals, voltages) |
| `analyze_task` | `target` | Inspect RTOS task state (priority, mutexes, blocking) |
| `run_diagnostic` | `target` | Run built-in peripheral/RTOS diagnostic |
| `submit_diagnosis` | `diagnosis`, `root_cause` | Record diagnosis before fixing |
| `apply_fix` | `fix_type`, `target`, ... | Apply a fix to resolve the issue |

### Example Actions
```json
{"action_type": "read_register", "target": "USART1", "register": "BRR"}
{"action_type": "write_register", "target": "USART1", "register": "BRR", "value": 417}
{"action_type": "apply_fix", "fix_type": "enable_priority_inheritance", "target": "spi_mutex"}
```

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `done` | bool | Episode ended |
| `reward` | float | Step reward [0.0, 1.0] |
| `message` | str | Human-readable result (register values, diagnostics, logs) |
| `data` | dict | Structured data for programmatic access |
| `system_status` | str | `fault` / `degraded` / `operational` |
| `error` | str? | Error if action was invalid |

## Tasks (5 total, easy → hard)

### 1. `uart_baud_mismatch` — Easy
**Symptom:** UART receiving corrupted data (CRC failures, overrun errors).

The STM32 communicates with an external sensor over USART1. The BRR register is misconfigured, producing 9600 baud instead of the required 19200. The agent must calculate the correct BRR value from the APB2 clock frequency and write it.

**Skills tested:** Register reading, baud rate calculation, clock tree understanding.

### 2. `i2c_sensor_failure` — Medium
**Symptom:** I2C sensor (IMU) not responding — NACK on every address attempt.

Three independent issues must be identified: wrong slave address (AD0 pin routing), I2C clock exceeding 400kHz spec, and GPIO output type set to push-pull instead of open-drain. Agent must fix at least 2 of 3.

**Skills tested:** Multi-factor diagnosis, I2C protocol knowledge, GPIO configuration.

### 3. `rtos_priority_inversion` — Hard
**Symptom:** High-priority sensor task intermittently misses deadlines.

A classic priority inversion: a low-priority task holds a mutex, gets preempted by a medium-priority task, blocking the high-priority task. The mutex was created as a binary semaphore (no priority inheritance). Agent must analyze task states and mutex ownership to identify the pattern.

**Skills tested:** RTOS scheduling, mutex semantics, concurrency debugging.

### 4. `dma_cache_coherency` — Hard
**Symptom:** CPU reads stale data from DMA buffer despite DMA transferring correctly.

On a Cortex-M7 with D-cache enabled, DMA writes to memory bypass the CPU cache. The MPU region is configured as write-back cacheable, so the CPU reads cached (stale) values. Agent must understand the memory hierarchy and either invalidate the cache or reconfigure the MPU.

**Skills tested:** Cache architecture, DMA operation, MPU configuration, memory-mapped I/O.

### 5. `watchdog_reset_loop` — Medium
**Symptom:** System stuck in boot loop, resetting every ~80ms.

The independent watchdog (IWDG) starts early in the boot sequence, but PLL lock + peripheral init takes longer than the configured timeout. Additionally, flash wait states are wrong for the 72MHz clock, causing further delays. Agent must calculate the watchdog timeout and fix the timing.

**Skills tested:** Watchdog timer configuration, boot sequence analysis, flash latency.

## Reward Function

Rewards are shaped across the full debugging trajectory:

| Component | Weight | Description |
|-----------|--------|-------------|
| Register exploration | 20% | Reading key diagnostic registers |
| Diagnosis accuracy | 25% | Correct identification of root cause |
| Fix correctness | 40% | Applying the right fix (structural grading) |
| Efficiency | 10% | Fewer steps = higher score |
| Penalties | -5% | Wrong fix attempts degrade score |

**Dynamic simulation features:**
- Correct register writes produce success logs and update related registers (e.g., ISR clears error flags)
- Wrong writes produce failure logs and can degrade system status
- Multiple wrong fix attempts increase penalty
- Grading is **structural** — based on actual register values written, not keyword matching

## Baseline Scores

Simulated optimal agent (reads key registers, submits diagnosis, applies correct fix):

| Task | Difficulty | Score | Steps |
|------|-----------|-------|-------|
| `uart_baud_mismatch` | Easy | 0.895 | 11 |
| `i2c_sensor_failure` | Medium | 0.865 | 13 |
| `rtos_priority_inversion` | Hard | 0.850 | 10 |
| `dma_cache_coherency` | Hard | 0.906 | 11 |
| `watchdog_reset_loop` | Medium | 0.869 | 13 |
| **Average** | | **0.877** | |

With zero investigation (just spamming read_log until step limit): all tasks score **0.000**.

## Setup & Usage

### Docker (recommended)
```bash
docker build -t firmware-debug-env .
docker run -p 7860:7860 firmware-debug-env
```

### Local Development
```bash
pip install -e ".[dev]"
uvicorn firmware_debug_env.server.app:app --host 0.0.0.0 --port 7860
```

### API Endpoints
```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Start a debugging session
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "uart_baud_mismatch"}'

# Execute a debugging action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "read_register", "target": "USART1", "register": "BRR"}}'

# Get current state
curl http://localhost:7860/state
```

### Run Inference
```bash
export HF_TOKEN="your-token"
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

## Technical Details

- **Framework:** OpenEnv (Meta PyTorch)
- **Server:** FastAPI + Uvicorn
- **Models:** Pydantic v2 with strict typing
- **Container:** Python 3.11 slim, runs on 2 vCPU / 8GB RAM
- **Inference runtime:** < 5 minutes per task, < 20 minutes total

## Author

**Ayush Kadali** — Firmware & Embedded Systems Engineer
- B.Tech CSE (AI & Data Science), MIT-WPU Pune
- Software Lead, CubeSat flight software (NASA cFS, STM32MP257)
- ISRO PSLV-C60 payload (STM32H7, FreeRTOS, 0.91 attitude correlation)
- Expertise: STM32, FreeRTOS, SPI/I2C/UART/DMA, sensor fusion, edge AI
