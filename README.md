# multi_robot_nav

# Multi-Robot RL Simulator

This project builds on top of the [irsim](https://github.com/hanruihua/ir-sim) simulator. The original simulator was cloned and extended to support reinforcement learning (RL) for multi-robot navigation tasks.

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Python Path

Add the `src` directory to your Python path:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

(Optional) Add this to your shell config file (e.g., `.bashrc` or `.zshrc`) for persistent access.

---

## 🏋️‍♂️ Training a Policy

1. Navigate to the training directory:

```bash
cd ./src/rl/policy_train/
```

2. Edit the `custom_env.yaml` file in this directory to configure the number of robots and other environment settings.
3. Start the training process:

```bash
python train_process.py
```

---

## 🧚‍♂️ Testing a Trained Policy

1. Navigate to the testing directory:

```bash
cd ./src/rl/policy_test/
```

2. Edit the `custom_env.yaml` file to match the desired test setup.
3. Inside `policy_test.py`, update the path to the trained model checkpoint.
4. Run the test:

```bash
python policy_test.py
```

---

## 📁 Directory Overview

```
src/
└── rl/
    ├── policy_train/
    │   ├── custom_env.yaml
    │   └── train_process.py
    └── policy_test/
        ├── custom_env.yaml
        └── policy_test.py
```

---
