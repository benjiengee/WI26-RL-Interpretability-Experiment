import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ICUResourceAllocationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_icu_beds=5, max_queue=10, episode_length=200):
        super().__init__()

        self.max_icu_beds = max_icu_beds
        self.max_queue = max_queue
        self.episode_length = episode_length

        # Actions: ADMIT, DELAY, TRANSFER, REJECT
        self.action_space = spaces.Discrete(4)

        # Observation space (normalized)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )

        self.reset()

    # -----------------------
    # Environment lifecycle
    # -----------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.timestep = 0
        self.available_icu_beds = self.max_icu_beds
        self.waiting_patients = []

        self._generate_patient()

        return self._get_observation(), {}

    def step(self, action):
        self.timestep += 1

        reward = 0.0
        terminated = False
        truncated = self.timestep >= self.episode_length

        patient = self.current_patient

        # --- Action effects ---
        if action == 0:  # ADMIT
            if self.available_icu_beds > 0:
                self.available_icu_beds -= 1
                reward += self._admit_patient(patient)
            else:
                reward -= 1.0  # tried to admit with no beds

        elif action == 1:  # DELAY
            reward += self._delay_patient(patient)

        elif action == 2:  # TRANSFER
            reward += self._transfer_patient(patient)

        elif action == 3:  # REJECT
            reward += self._reject_patient(patient)

        # ICU discharges
        self._process_icu_discharges()

        # New patient arrives
        self._generate_patient()

        obs = self._get_observation()
        info = {}

        return obs, reward, terminated, truncated, info

    # -----------------------
    # Patient generation
    # -----------------------

    def _generate_patient(self):
        """Create a new patient with latent state"""
        self.current_patient = {
            "severity": np.clip(np.random.beta(2, 2), 0, 1),
            "risk": np.clip(np.random.beta(2, 3), 0, 1),
            "los": np.random.randint(2, 15)
        }

    # -----------------------
    # Observation model
    # -----------------------

    def _get_observation(self):
        p = self.current_patient

        # Observable symptoms (noisy functions of latent state)
        heart_rate = np.clip(0.5 + 0.4 * p["severity"] + np.random.normal(0, 0.05), 0, 1)
        systolic_bp = np.clip(1.0 - 0.6 * p["severity"] + np.random.normal(0, 0.05), 0, 1)
        resp_rate = np.clip(0.4 + 0.5 * p["risk"] + np.random.normal(0, 0.05), 0, 1)
        oxygen_sat = np.clip(1.0 - 0.7 * p["severity"] + np.random.normal(0, 0.05), 0, 1)
        lactate = np.clip(0.3 + 0.6 * p["risk"] + np.random.normal(0, 0.05), 0, 1)
        temperature = np.clip(0.4 + 0.3 * p["risk"] + np.random.normal(0, 0.05), 0, 1)
        mental_status = np.clip(1.0 - 0.8 * p["severity"] + np.random.normal(0, 0.05), 0, 1)

        obs = np.array([
            heart_rate,
            systolic_bp,
            resp_rate,
            oxygen_sat,
            lactate,
            temperature,
            mental_status,
            self.available_icu_beds / self.max_icu_beds,
            len(self.waiting_patients) / self.max_queue
        ], dtype=np.float32)

        return obs

    # -----------------------
    # Action handlers
    # -----------------------

    def _admit_patient(self, patient):
        # Prevent deterioration
        benefit = 2.0 * patient["risk"]
        self.waiting_patients.append({
            "los": patient["los"]
        })
        return benefit

    def _delay_patient(self, patient):
        deterioration_prob = patient["risk"]
        if np.random.rand() < deterioration_prob:
            return -3.0
        return -0.2

    def _transfer_patient(self, patient):
        # Assume partial benefit
        return 0.5 * patient["risk"]

    def _reject_patient(self, patient):
        # Risky if severe
        return -2.0 * patient["severity"]

    # -----------------------
    # ICU dynamics
    # -----------------------

    def _process_icu_discharges(self):
        remaining = []
        for p in self.waiting_patients:
            p["los"] -= 1
            if p["los"] <= 0:
                self.available_icu_beds += 1
            else:
                remaining.append(p)
        self.waiting_patients = remaining

    # -----------------------
    # Rendering (optional)
    # -----------------------

    def render(self):
        print(
            f"Beds available: {self.available_icu_beds} | "
            f"ICU patients: {len(self.waiting_patients)}"
        )
