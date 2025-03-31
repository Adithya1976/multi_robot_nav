
# Multi-Robot RL Simulator

This project builds on top of the [irsim](https://github.com/robot-perception-group/irsim) simulator. The original simulator was cloned and extended to support reinforcement learning (RL) for multi-robot navigation tasks.

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Adithya1976/multi_robot_nav
cd multi_robot_nav
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

## 🛫 Using the BARN Dataset

1. Download the BARN dataset from the specified link (to be provided) and place it inside the `./src/barn/` directory. Rename the downloaded folder to `datasets`.
2. Navigate to the dataset script directory:

```bash
cd ./src/barn/
```

3. Run the notebook to generate the dataset:

```bash
jupyter notebook dataset.ipynb
```

4. Copy the generated dataset to both the `policy_train` and `policy_test` folders.
5. To train using the static BARN maps:

```bash
cd ./src/rl/policy_train/
python train_process_barn.py
```

6. To test a policy on BARN maps:

```bash
cd ./src/rl/policy_test/
python policy_test_barn.py
```

---

## 📁 Directory Overview

```
src/
├── barn/
│   └── dataset.ipynb
└── rl/
    ├── policy_train/
    │   ├── custom_env.yaml
    │   ├── train_process.py
    │   └── train_process_barn.py
    └── policy_test/
        ├── custom_env.yaml
        ├── policy_test.py
        └── policy_test_barn.py
```

---
