from __future__ import annotations

from typing import Any

from collections import deque
from itertools import combinations

import numpy as np
import random
from constants import (
    ACQUISITION_SESSION_TRIALS,
    ACTION_ENTER_WELL, ACTION_PAUSE,
    AGENT_VIEW_BEHIND, AGENT_VIEW_SIZE,
    BARRIER_LOCATIONS, CELL_VIEW_BEHIND,
    CORNERS, CUE_LOCATIONS,
    EMBEDDING_DIM, EMBEDDING_PARQUET_PATH, EYE_IMG_SIZE,
    EXPB_ACCLIMATION_STEPS, EXPB_BARRIER_DELAY_STEPS,
    EXPB_MAX_STEPS, EXPB_NUM_REWARDS,
    EXPOSURE_ITI_STEPS, EXPOSURE_ITI_STD,
    EXPOSURE_MAX_STEPS, EXPOSURE_NUM_REWARDS,
    GOAL_LOCATION_MAP, NUM_ARMS,
    PRETRIAL_MIN_STEPS, 
    PRETRIAL_START_MIN_STEPS,  # Integrated new constant
    PRETRIAL_TRIGGER_POSITIONS,
    RENDER_FPS,
    STATE_BASE, STATE_EXPA, STATE_EXPB, STATE_PRETRIAL, STATE_TRIAL, STATE_ITI,
    STEP_FORWARD_COST, STEP_TURN_COST,
    TRIGGER_LOCATIONS, VIEW_TILE_SIZE,
    WELL_EXIT_POSES, WELL_LOCATIONS, WELL_REWARD_SCR,
)
from trial_sequence_validation import (
    validate_sequence_start_only, validate_sequence_multi,
    shuffle_uniform_chunks, shuffle_acq_then_probe, shuffle_acq_then_novel,
)
from trial_sequence_gen import (
    get_f2_trained_pairs, get_f2_no_cue_pairs,
    get_f2_novel_route_probe_pairs, get_f2_reversal_probe_pairs,
    get_f2_rotate_pairs,
    get_f1_trained_pairs, get_f1_no_cue_pairs,
    get_f1_novel_route_probe_pairs, get_f1_reversal_probe_pairs,
    get_f1_rotate_pairs,
    get_pi_novel_route_no_cue_probe_pairs, get_pi_reversal_no_cue_probe_pairs,
    get_vc_novel_route_rotate_probe_pairs, get_vc_reversal_rotate_probe_pairs,
)

from gymnasium import spaces
from gymnasium.core import ObsType

from minigrid.core.constants import COLORS, COLOR_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Ball, Floor
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_rect

import pandas as pd

# CONDITIONAL LOOKUP DICTIONARIES
DEFAULT_AGENT_START_POS = (11, 1)
DEFAULT_AGENT_START_DIR = 1

CORNER_LEFT_TURN_WELL_EXIT = {
    (10, 2): ((10, 3), 1),
    (10, 10): ((9, 10), 2),
    (2, 10): ((2, 9), 3),
    (2, 2): ((3, 2), 0),
}

WELL_EXIT_FORWARD = {
    (11, 1): ((10, 2), 2),
    (11, 11): ((10, 10), 3),
    (1, 11): ((2, 10), 0),
    (1, 1): ((2, 2), 1),
}

WELL_ENTRY_PICKUP = {
    (10, 10): ((11, 11), 1),
    (2, 10): ((1, 11), 2),
    (2, 2): ((1, 1), 3),
    (10, 2): ((11, 1), 0),
}

TURN_ONE_SENW_MAP = {
    (7, 6): 1,
    (5, 6): 0,
    (6, 5): 1,
    (6, 7): 0,
}

TURN_ONE_NESW_MAP = {
    (7, 6): 0,
    (5, 6): 1,
    (6, 5): 0,
    (6, 7): 1,
}

TURN_TWO_SET_A = {(10, 7), (2, 5), (5, 2), (7, 10)}
TURN_TWO_SET_B = {(10, 5), (2, 7), (7, 2), (5, 10)}

EXPB_BARRIER_SEQUENCE = [
    ('expb_x_x_xx',    'timed', None,   EXPB_ACCLIMATION_STEPS),
    ('expb_exxx_x_xx', 'timed', None,   EXPB_BARRIER_DELAY_STEPS),
    ('expb_enxx_x_xx', 'zone',  (8, 6), EXPB_BARRIER_DELAY_STEPS),
    ('expb_enwx_x_xx', 'zone',  (6, 4), EXPB_BARRIER_DELAY_STEPS),
    ('expb_enws_x_xx', 'zone',  (4, 6), EXPB_BARRIER_DELAY_STEPS),
    ('expb_xxxx_x_xx', 'zone',  (6, 8), 0),
]

# REGION ######################### BEGIN ENVIRONMENT CODE ###############################

COLORS["cue_on_rgb"] = np.array([255, 0, 255])
COLORS["cue_off_rgb"] = np.array([25, 0, 255])
COLORS["chasm_rgb"] = np.array([0, 0, 255])
COLORS["wall_rgb"] = np.array([0, 255, 0])
COLORS["black"] = np.array([0, 0, 0])
COLOR_TO_IDX["cue_on_rgb"] = 6
COLOR_TO_IDX["cue_off_rgb"] = 7
COLOR_TO_IDX["chasm_rgb"] = 8
COLOR_TO_IDX["wall_rgb"] = 9
COLOR_TO_IDX["black"] = 10   

class Chasm(Wall):
    def __init__(self, color='chasm_rgb'):
        super().__init__(color)
    def see_behind(self):
        return True
    
class Barrier(Wall):
    def __init__(self, color='wall_rgb'):
        super().__init__(color)
    def see_behind(self):
        return True

class Well(Ball):
    def __init__(self, color='black', visible=False, has_reward=False):
        super().__init__(color)
        self.visible = visible
        self.has_reward = has_reward
    def can_toggle(self):
        return True
    def can_overlap(self):
        return True
    def render(self, img):
        if self.visible:
            c = COLORS[self.color]
            fill_coords(img, point_in_rect(0, 1, 0, 1), c)

class Trigger(Floor):
    def __init__(self, color='black', visible=False, trigger_type=None):
        super().__init__(color)
        self.visible = visible
        self.color = color
        self.trigger_type = trigger_type
        if trigger_type not in ('A', 'B', 'S', None):
            raise ValueError("trigger_type must be 'A', 'B', 'S', or None")
    def get_trigger_type(self):
        return self.trigger_type
    def render(self, img):
        if self.visible:
            c = COLORS[self.color]
            fill_coords(img, point_in_rect(0, 1, 0, 1), c)

class CornerMazeEnv(MiniGridEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}
    def __init__(
        self,
        size=13,
        agent_start_pos=DEFAULT_AGENT_START_POS,
        agent_start_dir=DEFAULT_AGENT_START_DIR,
        max_steps: int | None = None,
        session_type: str | None = None,
        agent_cue_goal_orientation: str | None = None,
        start_goal_location: str | None = None,
        obs_mode: str = "view",
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            if session_type == 'exposure':
                max_steps = EXPOSURE_MAX_STEPS
            elif session_type == 'exposure_b':
                max_steps = EXPB_MAX_STEPS
            else:
                max_steps = 4 * size**2

        self.session_type = session_type
        self.agent_cue_goal_orientation = agent_cue_goal_orientation
        self.start_goal_location = start_goal_location

        if obs_mode not in ("view", "embedding"):
            raise ValueError(f"obs_mode must be 'view' or 'embedding', got '{obs_mode}'")
        self.obs_mode = obs_mode

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        if self.obs_mode == "view":
            self.observation_space = spaces.Dict({
                "image": spaces.Box(
                    low=0, high=255, shape=(AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, 3), dtype=np.uint8
                ),
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            })
        else:
            self.observation_space = spaces.Dict({
                "embedding": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(EMBEDDING_DIM,), dtype=np.float64
                ),
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            })

        self.reward_range = (-1, 1)
        self.init_variables()
    
    def init_variables(self):
        self.maze_state_array = [STATE_BASE, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts = {}
        self.layouts['x_x_xx'] = [STATE_BASE, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        
        well_names = ['se', 'sw', 'nw', 'ne']
        for r in range(1, 5):
            for combo in combinations(range(4), r):
                name = 'expa_x_x_' + '_'.join(well_names[i] for i in combo)
                layout = [STATE_EXPA, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
                for i in combo:
                    layout[21 + i] = 1
                self.layouts[name] = layout

        self.layouts['expb_x_x_xx']    = [STATE_EXPB, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 1,1,1,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['expb_exxx_x_xx'] = [STATE_EXPB, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,1,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['expb_enxx_x_xx'] = [STATE_EXPB, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['expb_enwx_x_xx'] = [STATE_EXPB, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['expb_enws_x_xx'] = [STATE_EXPB, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['expb_xxxx_x_xx'] = [STATE_EXPB, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]

        start_arms = ['n', 'e', 's', 'w']
        cues = ['n', 'e', 's', 'w', 'x']
        goals = ['ne', 'se', 'sw', 'nw']

        base_trl_layouts = {
            'n' : [STATE_TRIAL, 0,0,0, 1,0,1, 0,0,0, 1,0,1, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0],
            'e' : [STATE_TRIAL, 1,0,1, 0,0,0, 1,0,1, 0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0],
            's' : [STATE_TRIAL, 0,0,0, 1,0,1, 0,0,0, 1,0,1, 0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0],
            'w' : [STATE_TRIAL, 1,0,1, 0,0,0, 1,0,1, 0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        }
        for start_arm in start_arms:
            for cue in cues:
                for goal in goals:
                    variable_name = f'trl_{start_arm}_{cue}_{goal}'
                    layout = base_trl_layouts[start_arm].copy()
                    layout[17] = 1 if cue == 'e' else 0
                    layout[18] = 1 if cue == 's' else 0
                    layout[19] = 1 if cue == 'w' else 0
                    layout[20] = 1 if cue == 'n' else 0
                    layout[21] = 1 if goal == 'se' else 0
                    layout[22] = 1 if goal == 'sw' else 0
                    layout[23] = 1 if goal == 'nw' else 0
                    layout[24] = 1 if goal == 'ne' else 0
                    self.layouts[variable_name] = layout

        self.layouts['iti_e_x_xx'] = [STATE_ITI, 0,0,0, 0,1,0, 0,1,0, 0,1,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 2,0, 0,0, 0,1, 0,0,0,0]
        self.layouts['iti_e_x_ne'] = [STATE_ITI, 0,0,1, 0,1,0, 0,1,0, 1,1,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 3,0,0,0]
        self.layouts['iti_e_x_se'] = [STATE_ITI, 1,0,0, 0,1,1, 0,1,0, 0,1,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 3,0,0,0]
        self.layouts['iti_s_x_xx'] = [STATE_ITI, 0,1,0, 0,0,0, 0,1,0, 0,1,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,1, 0,0, 2,0, 0,0, 0,0,0,0]
        self.layouts['iti_s_x_se'] = [STATE_ITI, 1,1,0, 0,0,1, 0,1,0, 0,1,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,3,0,0]
        self.layouts['iti_s_x_sw'] = [STATE_ITI, 0,1,0, 1,0,0, 0,1,1, 0,1,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,3,0,0]
        self.layouts['iti_w_x_xx'] = [STATE_ITI, 0,1,0, 0,1,0, 0,0,0, 0,1,0, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,1, 0,0, 2,0, 0,0,0,0]
        self.layouts['iti_w_x_sw'] = [STATE_ITI, 0,1,0, 1,1,0, 0,0,1, 0,1,0, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,3,0]
        self.layouts['iti_w_x_nw'] = [STATE_ITI, 0,1,0, 0,1,0, 1,0,0, 0,1,1, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,3,0]
        self.layouts['iti_n_x_xx'] = [STATE_ITI, 0,1,0, 0,1,0, 0,1,0, 0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0, 2,0, 0,0, 0,1, 0,0, 0,0,0,0]
        self.layouts['iti_n_x_nw'] = [STATE_ITI, 0,1,0, 0,1,0, 1,1,0, 0,0,1, 0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,3]
        self.layouts['iti_n_x_ne'] = [STATE_ITI, 0,1,1, 0,1,0, 0,1,0, 1,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,3]

        base_pre_layouts = {
            'n' : [STATE_PRETRIAL, 0,0,0, 1,0,1, 0,0,0, 1,0,1, 0,1,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,4],
            'e' : [STATE_PRETRIAL, 1,0,1, 0,0,0, 1,0,1, 0,0,0, 1,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 4,0,0,0],
            's' : [STATE_PRETRIAL, 0,0,0, 1,0,1, 0,0,0, 1,0,1, 0,1,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,4,0,0],
            'w' : [STATE_PRETRIAL, 1,0,1, 0,0,0, 1,0,1, 0,0,0, 1,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,4,0],
        }
        for start_arm in start_arms:
            for cue in cues:
                variable_name = f'pre_{start_arm}_{cue}_xx'
                layout = base_pre_layouts[start_arm].copy()
                layout[17] = 1 if cue == 'e' else 0
                layout[18] = 1 if cue == 's' else 0
                layout[19] = 1 if cue == 'w' else 0
                layout[20] = 1 if cue == 'n' else 0
                self.layouts[variable_name] = layout

        self.layouts.update({k: tuple(v) for k, v in self.layouts.items()})
        self.layout_name_lookup = {v: k for k, v in self.layouts.items()}

        if self.obs_mode == "embedding" or getattr(self, 'render_mode', None) == "human":
            self._load_embeddings()

        self.maze_config_expa_lookup = {}
        for r in range(1, 5):
            for combo in combinations(range(4), r):
                name = 'expa_x_x_' + '_'.join(well_names[i] for i in combo)
                self.maze_config_expa_lookup[frozenset(combo)] = self.layouts[name]

        self.maze_config_trl_list = [[[None for _ in goals] for _ in cues] for _ in start_arms]

        for i, startarm in enumerate(start_arms):
            for j, cue in enumerate(cues):
                for k, goal in enumerate(goals):
                    layout_name = f'trl_{startarm}_{cue}_{goal}'
                    self.maze_config_trl_list[i][j][k] = self.layouts.get(layout_name)

        self.maze_config_iti_list = [[None for _ in range(3)] for _ in range(4)]
        self.maze_config_iti_list[0][0] = self.layouts.get('iti_n_x_xx')
        self.maze_config_iti_list[0][1] = self.layouts.get('iti_n_x_nw')
        self.maze_config_iti_list[0][2] = self.layouts.get('iti_n_x_ne')
        self.maze_config_iti_list[1][0] = self.layouts.get('iti_e_x_xx')
        self.maze_config_iti_list[1][1] = self.layouts.get('iti_e_x_ne')
        self.maze_config_iti_list[1][2] = self.layouts.get('iti_e_x_se')
        self.maze_config_iti_list[2][0] = self.layouts.get('iti_s_x_xx')
        self.maze_config_iti_list[2][1] = self.layouts.get('iti_s_x_se')
        self.maze_config_iti_list[2][2] = self.layouts.get('iti_s_x_sw')
        self.maze_config_iti_list[3][0] = self.layouts.get('iti_w_x_xx')
        self.maze_config_iti_list[3][1] = self.layouts.get('iti_w_x_sw')
        self.maze_config_iti_list[3][2] = self.layouts.get('iti_w_x_nw')

        self.maze_config_pre_list = [[None for _ in cues] for _ in start_arms]
        for i, startarm in enumerate(start_arms):
            for j, cue in enumerate(cues):
                layout_name = f'pre_{startarm}_{cue}_xx'
                self.maze_config_pre_list[i][j] = self.layouts.get(layout_name)

        self.agent_pov_pos = ((AGENT_VIEW_SIZE // 2), (AGENT_VIEW_SIZE - 1) - AGENT_VIEW_BEHIND)
        visual_exclusion = []  
        for i in range(AGENT_VIEW_BEHIND):
            if i == 0:
                visual_exclusion.append(((((AGENT_VIEW_SIZE) - AGENT_VIEW_BEHIND)+ i),AGENT_VIEW_SIZE // 2))
            else:
                for j in range(i+1):
                    if j == 0:
                        visual_exclusion.append(((((AGENT_VIEW_SIZE) - AGENT_VIEW_BEHIND) + i), AGENT_VIEW_SIZE // 2))
                    else:
                        visual_exclusion.append(((((AGENT_VIEW_SIZE) - AGENT_VIEW_BEHIND) + i), AGENT_VIEW_SIZE // 2 - j))
                        visual_exclusion.append(((((AGENT_VIEW_SIZE) - AGENT_VIEW_BEHIND) + i), AGENT_VIEW_SIZE // 2 + j))
        self.visual_mask = np.ones((AGENT_VIEW_SIZE, AGENT_VIEW_SIZE), dtype=bool)
        for cell in visual_exclusion:
            self.visual_mask[cell[0], cell[1]] = False
        self.visual_mask = self.expand_matrix(self.visual_mask, VIEW_TILE_SIZE)
        
        wall_ledge_inclusion = []
        wall_ledge_inclusion.append((AGENT_VIEW_SIZE - AGENT_VIEW_BEHIND - 2, AGENT_VIEW_SIZE // 2))
        wall_ledge_inclusion.extend([(AGENT_VIEW_SIZE - AGENT_VIEW_BEHIND - 1, AGENT_VIEW_SIZE // 2 - 1),
                                    (AGENT_VIEW_SIZE - AGENT_VIEW_BEHIND - 1, AGENT_VIEW_SIZE // 2),
                                    (AGENT_VIEW_SIZE - AGENT_VIEW_BEHIND - 1, AGENT_VIEW_SIZE // 2 + 1)])
        wall_ledge_inclusion.append((AGENT_VIEW_SIZE - AGENT_VIEW_BEHIND, AGENT_VIEW_SIZE // 2))
        self.wall_ledge_mask = np.ones((AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, 2), dtype=bool)
        for cell in wall_ledge_inclusion:
            self.wall_ledge_mask[cell[0], cell[1], :] = False
        self.wall_ledge_mask = self.expand_matrix(self.wall_ledge_mask, VIEW_TILE_SIZE)
        
        self.action_space = spaces.Discrete(5)
        self.last_pose  = (None, None, None)
        self.trajectory = []
        self.turn_score = [None, None]
        self.trial_score = None
        self.session_reward = 0
        self.session_num_trials = None
        self.trial_count = None
        self.episode_trial_scores = []
        self.episode_turn_scores = []
        self.episode_scores = []
        self.phase_step_count = None
        self.session_phase = None
        self.iti_type = None
        self.pretrial_step_count = 0
        self.exposure_wells_remaining = None
        self.exposure_iti_counter = None
        self.exposure_iti_threshold = None
        self.exposure_reward_count = None
        self.expb_barrier_step = None
        self.expb_delay_counter = 0
        self.expb_delay_target = 0
        self.expb_waiting_for_zone = False
        self.expb_target_zone = None
        self.episode = 0
        self.episode_data_rows = []
        self.fwd_pos = None
        self.fwd_cell = None
        self.cur_cell = None
        self.pseudo_session_score = deque([0] * ACQUISITION_SESSION_TRIALS, maxlen=ACQUISITION_SESSION_TRIALS)

    def expand_matrix(self, original_matrix, scale_factor):
        original_shape = original_matrix.shape
        original_height, original_width = original_shape[:2]
        new_height = original_height * scale_factor
        new_width = original_width * scale_factor

        if len(original_shape) == 2:
            expanded_matrix = np.zeros((new_height, new_width), dtype=original_matrix.dtype)
            for i in range(original_height):
                for j in range(original_width):
                    expanded_matrix[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor] = original_matrix[i, j]
        elif len(original_shape) == 3:
            channels = original_shape[2]
            expanded_matrix = np.zeros((new_height, new_width, channels), dtype=original_matrix.dtype)
            for i in range(original_height):
                for j in range(original_width):
                    expanded_matrix[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor, :] = original_matrix[i, j, :]
        return expanded_matrix

    def _load_embeddings(self):
        import os
        parquet_path = os.path.join(os.path.dirname(__file__), EMBEDDING_PARQUET_PATH)
        emb_df = pd.read_parquet(parquet_path)
        self._pose_to_embedding = {}
        self._pose_to_left_eye = {}
        self._pose_to_right_eye = {}
        for _, row in emb_df.iterrows():
            emb_vec = np.array(row['embedding'], dtype=np.float64)
            left_img = np.stack(row['left_eye_img']).astype(np.float64)
            right_img = np.stack(row['right_eye_img']).astype(np.float64)
            for pose_label in row['poses']:
                self._pose_to_embedding[pose_label] = emb_vec
                self._pose_to_left_eye[pose_label] = left_img
                self._pose_to_right_eye[pose_label] = right_img
        self._zero_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        self._zero_eye = np.zeros((EYE_IMG_SIZE, EYE_IMG_SIZE), dtype=np.float64)

    def _get_pose_label(self) -> str:
        x, y = self.agent_pos
        d = self.agent_dir
        seq_entry = self.grid_configuration_sequence[self.sequence_count]
        if isinstance(seq_entry, tuple):
            layout_name = self.layout_name_lookup.get(seq_entry, '')
        elif isinstance(seq_entry, list):
            idx = getattr(self, '_iti_config_idx', 0)
            layout_name = self.layout_name_lookup.get(seq_entry[idx], '') if seq_entry else ''
        else:
            layout_name = ''
        if not layout_name:
            return f"x_x_x_xx_{x}_{y}_{d}"
        parts = layout_name.split('_')
        phase = parts[0]
        if phase == 'expa':
            prefix = 'expa_x_x_xx'
        elif phase in ('trl', 'pre'):
            arm = parts[1]
            cue = parts[2]
            prefix = f'{phase}_{arm}_{cue}_xx'
        else:
            prefix = layout_name
        return f'{prefix}_{x}_{y}_{d}'

    def _build_observation(self) -> dict:
        if self.obs_mode == "view":
            img = self.get_pov_render_mod(tile_size=VIEW_TILE_SIZE)
            return {'image': img, 'direction': self.agent_dir, 'mission': self.mission}
        pose_label = self._get_pose_label()
        emb = self._pose_to_embedding.get(pose_label, self._zero_embedding)
        return {'embedding': emb, 'direction': self.agent_dir, 'mission': self.mission}

    def put_obj_rect(self, obj, topX, topY, width, height):
        for x in range(topX, topX + width):
            for y in range(topY, topY + height):
                self.grid.set(x, y, obj)

    def update_grid_configuration(self, grid_configuration: tuple[int, ...]) -> None:
        for i, bl in enumerate(BARRIER_LOCATIONS):
            if grid_configuration[i+1] != self.maze_state_array[i+1]:
                if grid_configuration[i+1] == 1:
                    self.put_obj(Barrier(), bl[0], bl[1])
                    self.maze_state_array[i+1] = 1
                else:
                    self.grid.set(bl[0], bl[1], None)
                    self.maze_state_array[i+1] = 0    
        for i, cl in enumerate(CUE_LOCATIONS):
            if grid_configuration[i+17] != self.maze_state_array[i+17]:
                if grid_configuration[i+17] == 1:
                    self.put_obj(Wall(color='cue_on_rgb'), cl[0], cl[1])
                    self.maze_state_array[i+17] = 1
                else:
                    self.put_obj(Wall(color='cue_off_rgb'), cl[0], cl[1])
                    self.maze_state_array[i+17] = 0
        for i, wl in enumerate(WELL_LOCATIONS):
            if grid_configuration[i+21] != self.maze_state_array[i+21]:
                if grid_configuration[i+21] == 1:
                    self.put_obj(Well(has_reward=True), wl[0], wl[1])
                    self.maze_state_array[i+21] = 1
                else:
                    self.put_obj(Well(), wl[0], wl[1])
                    self.maze_state_array[i+21] = 0
        for i, tl in enumerate(TRIGGER_LOCATIONS):
            if grid_configuration[i+25] != self.maze_state_array[i+25]:
                if grid_configuration[i+25] == 1:
                    self.put_obj(Trigger(trigger_type='A'), tl[0], tl[1])
                    self.maze_state_array[i+25] = 1
                elif grid_configuration[i+25] == 2:
                    self.put_obj(Trigger(trigger_type='B'), tl[0], tl[1])
                    self.maze_state_array[i+25] = 2
                elif grid_configuration[i+25] == 3:
                    self.put_obj(Trigger(trigger_type='S'), tl[0], tl[1])
                    self.maze_state_array[i+25] = 3
                else:
                    self.grid.set(tl[0], tl[1], None)
                    self.maze_state_array[i+25] = 0
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.turn_score = [None, None]
        self.trial_score = None
        self.session_phase = None
        self.iti_type = None
        self.episode_trial_scores = []
        self.episode_turn_scores = []
        self.session_num_trials = None
        self.trial_count = 0
        self.sequence_count = 0
        self.phase_step_count = 0
        self.pretrial_step_count = 0
        self.session_reward = 0
        self.exposure_wells_remaining = set(range(4))
        self.exposure_iti_threshold = int(random.gauss(EXPOSURE_ITI_STEPS, EXPOSURE_ITI_STD))
        self.exposure_iti_counter = self.exposure_iti_threshold
        self.exposure_reward_count = 0
        if self.session_type == 'exposure_b':
            self.expb_barrier_step = 0
            self.expb_delay_counter = 0
            self.expb_delay_target = EXPB_ACCLIMATION_STEPS
            self.expb_waiting_for_zone = False
            self.expb_target_zone = None
        else:
            self.expb_barrier_step = None
        self.trajectory = []
        self.fwd_pos = None
        self.fwd_cell = None
        self.cur_cell = None
        self._gen_grid(self.width, self.height)
        assert (self.agent_pos >= (0, 0) if isinstance(self.agent_pos, tuple) else all(self.agent_pos >= 0) and self.agent_dir >= 0)
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()
        self.step_count = 0
        if self.render_mode == "human":
            self.render()
        obs_mod = self._build_observation()
        info = {}
        return obs_mod, info

    def gen_grid_configuration_sequence(self) -> tuple[list, int]:
        goal_location_index = GOAL_LOCATION_MAP.get(self.start_goal_location, random.randint(0, NUM_ARMS - 1))
        ori = self.agent_cue_goal_orientation
        gli = goal_location_index
        def _gen_simple_pool(pool, chunk_size, threepeat_limit):
            while True:
                result = shuffle_uniform_chunks(pool, chunk_size)
                if validate_sequence_start_only(result, threepeat_limit):
                    return [(s, c, g) for s, c, r, g in result]
        def _gen_multi_pool(pool, chunk_size, start_limit=3, route_limit=3, goal_limit=3):
            while True:
                result = shuffle_uniform_chunks(pool, chunk_size)
                if validate_sequence_multi(result, start_limit, route_limit, goal_limit):
                    return [(s, c, g) for s, c, r, g in result]
        def _gen_novel_route(trained_pairs, probe_pairs, chunk_size, threepeat_limit=0):
            trained = list(trained_pairs); probe = list(probe_pairs)
            while True:
                result, success = shuffle_acq_then_novel(trained, probe, chunk_size)
                if not success: continue
                if validate_sequence_start_only(result, threepeat_limit):
                    return [(s, c, g) for s, c, r, g in result]
        def _gen_novel_route_multi(trained_pairs, probe_pairs, chunk_size, start_limit=3, route_limit=3, goal_limit=3):
            trained = list(trained_pairs); probe = list(probe_pairs)
            while True:
                result, success = shuffle_acq_then_novel(trained, probe, chunk_size)
                if not success: continue
                if validate_sequence_multi(result, start_limit, route_limit, goal_limit):
                    return [(s, c, g) for s, c, r, g in result]
        def _gen_reversal(trained_pairs, probe_pairs, chunk_size, threepeat_limit=0):
            trained = list(trained_pairs); probe = list(probe_pairs)
            while True:
                result = shuffle_acq_then_probe(trained, probe, chunk_size)
                if validate_sequence_start_only(result, threepeat_limit):
                    return [(s, c, g) for s, c, r, g in result]
        def _gen_reversal_multi(trained_pairs, probe_pairs, chunk_size, start_limit=3, route_limit=3, goal_limit=3):
            trained = list(trained_pairs); probe = list(probe_pairs)
            while True:
                result = shuffle_acq_then_probe(trained, probe, chunk_size)
                if validate_sequence_multi(result, start_limit, route_limit, goal_limit):
                    return [(s, c, g) for s, c, r, g in result]
        def gen_pi_vc_f2_single_trial(): return [(random.choice([1, 3]), 0, gli)]
        def gen_pi_vc_f2_acq():
            pool = get_f2_trained_pairs(ori, gli) * 4
            return _gen_simple_pool(pool, chunk_size=4, threepeat_limit=2)
        def gen_pi_vc_f2_novel_route():
            trained = get_f2_trained_pairs(ori, gli) * 4; probe = get_f2_novel_route_probe_pairs(ori, gli)
            return _gen_novel_route(trained, probe, chunk_size=6)
        def gen_pi_vc_f2_no_cue():
            pool = get_f2_no_cue_pairs(ori, gli) * 4
            return _gen_simple_pool(pool, chunk_size=4, threepeat_limit=2)
        def gen_pi_vc_f2_rotate(): pool = get_f2_rotate_pairs(ori); return _gen_multi_pool(pool, chunk_size=2)
        def gen_pi_vc_f2_reversal():
            trained = get_f2_trained_pairs(ori, gli) * 4; probe = get_f2_reversal_probe_pairs(ori, gli) * 4
            return _gen_reversal(trained, probe, chunk_size=10)
        def gen_pi_vc_f1_acq():
            pool = get_f1_trained_pairs(ori, gli) * 4
            return _gen_simple_pool(pool, chunk_size=4, threepeat_limit=2)
        def gen_pi_vc_f1_novel_route():
            trained = get_f1_trained_pairs(ori, gli) * 4; probe = get_f1_novel_route_probe_pairs(ori, gli)
            return _gen_novel_route(trained, probe, chunk_size=6)
        def gen_pi_vc_f1_no_cue():
            pool = get_f1_no_cue_pairs(ori, gli) * 4
            return _gen_simple_pool(pool, chunk_size=4, threepeat_limit=2)
        def gen_pi_vc_f1_rotate(): pool = get_f1_rotate_pairs(ori); return _gen_multi_pool(pool, chunk_size=2)
        def gen_pi_vc_f1_reversal():
            trained = get_f1_trained_pairs(ori, gli) * 4; probe = get_f1_reversal_probe_pairs(ori, gli) * 4
            return _gen_reversal(trained, probe, chunk_size=10)
        def gen_pi_novel_route_no_cue():
            trained = get_f2_no_cue_pairs(ori, gli) * 4; probe = get_pi_novel_route_no_cue_probe_pairs(ori, gli)
            return _gen_novel_route(trained, probe, chunk_size=6)
        def gen_pi_reversal_no_cue():
            trained = get_f2_no_cue_pairs(ori, gli) * 4; probe = get_pi_reversal_no_cue_probe_pairs(ori, gli) * 4
            return _gen_reversal(trained, probe, chunk_size=10)
        def gen_vc_acquisition(): pool = get_f2_rotate_pairs(ori); return _gen_multi_pool(pool, chunk_size=4)
        def gen_vc_novel_route_rotate():
            trained = get_f2_rotate_pairs(ori); probe = get_vc_novel_route_rotate_probe_pairs(ori)
            return _gen_novel_route_multi(trained, probe, chunk_size=3)
        def gen_vc_reversal_rotate():
            trained = get_f2_rotate_pairs(ori); probe = get_vc_reversal_rotate_probe_pairs(ori)
            return _gen_reversal_multi(trained, probe, chunk_size=10)
        def gen_exposure(): return []
        def gen_exposure_b(): return []

        SESSION_GENERATORS = {
            'exposure': gen_exposure, 'exposure_b': gen_exposure_b, 'PI+VC f2 single trial': gen_pi_vc_f2_single_trial,
            'PI+VC f2 acquisition': gen_pi_vc_f2_acq, 'PI+VC f2 novel route': gen_pi_vc_f2_novel_route,
            'PI+VC f2 no cue': gen_pi_vc_f2_no_cue, 'PI+VC f2 rotate': gen_pi_vc_f2_rotate,
            'PI+VC f2 reversal': gen_pi_vc_f2_reversal, 'PI+VC f1 acquisition': gen_pi_vc_f1_acq,
            'PI+VC f1 novel route': gen_pi_vc_f1_novel_route, 'PI+VC f1 no cue': gen_pi_vc_f1_no_cue,
            'PI+VC f1 rotate': gen_pi_vc_f1_rotate, 'PI+VC f1 reversal': gen_pi_vc_f1_reversal,
            'PI acquisition': gen_pi_vc_f2_no_cue, 'PI novel route no cue': gen_pi_novel_route_no_cue,
            'PI novel route cue': gen_pi_vc_f2_novel_route, 'PI reversal no cue': gen_pi_reversal_no_cue,
            'PI reversal cue': gen_pi_vc_f2_reversal, 'VC acquisition': gen_vc_acquisition,
            'VC novel route fixed': gen_pi_vc_f2_novel_route, 'VC novel route rotate': gen_vc_novel_route_rotate,
            'VC reversal fixed': gen_pi_vc_f2_reversal, 'VC reversal rotate': gen_vc_reversal_rotate,
        }
        generator = SESSION_GENERATORS.get(self.session_type)
        if generator is None: print('Invalid session type'); start_goal_cue_list = None
        else: start_goal_cue_list = generator()

        if self.session_type == 'exposure':
            all_wells = self.maze_config_expa_lookup[frozenset(range(4))]
            return [all_wells], EXPOSURE_NUM_REWARDS
        if self.session_type == 'exposure_b':
            initial_layout = self.layouts['expb_x_x_xx']
            return [initial_layout], EXPB_NUM_REWARDS

        grid_configuration_sequence = []
        len_sgc = len(start_goal_cue_list)
        num_trials = len_sgc
        for i, sgc in enumerate(start_goal_cue_list):
            start_arm = sgc[0]; cue = sgc[1]; goal = sgc[2]
            if len_sgc == 1:
                grid_configuration_sequence.append(self.maze_config_pre_list[start_arm][cue])
                grid_configuration_sequence.append(self.maze_config_trl_list[start_arm][cue][goal])
            elif len_sgc > 1:
                if i < len_sgc - 1: next_start_arm = start_goal_cue_list[i + 1][0]
                grid_configuration_sequence.append(self.maze_config_pre_list[start_arm][cue])
                grid_configuration_sequence.append(self.maze_config_trl_list[start_arm][cue][goal])
                if i < len_sgc - 1: grid_configuration_sequence.append(self.maze_config_iti_list[next_start_arm])
            else: grid_configuration_sequence = None
        return grid_configuration_sequence, num_trials

    def gen_start_pose(self) -> tuple[tuple[int, int], int]:
        first_config = self.grid_configuration_sequence[0]
        layout_name = self.layout_name_lookup.get(first_config, '')
        parts = layout_name.split('_')
        if len(parts) >= 2:
            if parts[0] == 'expa': return (6, 6), DEFAULT_AGENT_START_DIR
            if parts[0] == 'expb': return (6, 6), random.randint(0, 3)
            arm = parts[1]
            arm_to_pose = {
                'n': ((6, 3), 1), 'e': ((9, 6), 2), 's': ((6, 9), 3), 'w': ((3, 6), 0),
            }
            if arm in arm_to_pose: return arm_to_pose[arm]
        if self.grid_configuration_sequence[0][13] == 1: return (3, 6), 0
        elif self.grid_configuration_sequence[0][14] == 1: return (6, 3), 1
        elif self.grid_configuration_sequence[0][15] == 1: return (9, 6), 2
        elif self.grid_configuration_sequence[0][16] == 1: return (6, 9), 3
        else: return DEFAULT_AGENT_START_POS, DEFAULT_AGENT_START_DIR

    def is_agent_on_obj(self, obj):
        x, y = self.agent_pos
        obj_at_pos = self.grid.get(x, y)
        return isinstance(obj_at_pos, obj)

    def gen_obs_grid_mod(self, agent_view_size=None):
        topX, topY, botX, botY = self.get_view_exts(agent_view_size)
        if self.agent_dir == 0: topX -= CELL_VIEW_BEHIND
        elif self.agent_dir == 1: topY -= CELL_VIEW_BEHIND
        elif self.agent_dir == 2: topX += CELL_VIEW_BEHIND
        elif self.agent_dir == 3: topY += CELL_VIEW_BEHIND
        agent_view_size = agent_view_size or self.agent_view_size
        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)
        for i in range(self.agent_dir + 1): grid = grid.rotate_left()
        return grid
    
    def get_pov_render_mod(self, tile_size):
        grid = self.gen_obs_grid_mod(AGENT_VIEW_SIZE)
        img = grid.render(tile_size, agent_pos = self.agent_pov_pos, agent_dir=3, highlight_mask=None)
        img[img == 100] = 0   
        visual_condition = (img[:, :, 0] == 255) | (img[:, :, 0] == 25)
        img[:,:,0][~self.visual_mask & visual_condition] = 0
        img[:, :, 1][self.wall_ledge_mask[:, :, 0]] = 0
        img[:, :, 2][self.wall_ledge_mask[:, :, 1]] = 0
        return img
    
    def get_allocentric_frame(self, tile_size=32):
        return self.grid.render(tile_size=tile_size, agent_pos=self.agent_pos, agent_dir=self.agent_dir, highlight_mask=None)

    def render(self):
        if self.render_mode != "human" or not hasattr(self, '_pose_to_left_eye') or not hasattr(self, 'sequence_count'):
            return super().render()
        import pygame
        import pygame.freetype
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)
        img = np.transpose(img, axes=(1, 0, 2))
        if self.render_size is None: self.render_size = img.shape[:2]
        pose_label = self._get_pose_label()
        left_eye = self._pose_to_left_eye.get(pose_label, self._zero_eye)
        right_eye = self._pose_to_right_eye.get(pose_label, self._zero_eye)
        left_rgb = np.repeat((left_eye * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
        right_rgb = np.repeat((right_eye * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
        grid_surf = pygame.surfarray.make_surface(img)
        grid_size = grid_surf.get_size()
        eye_display_h = grid_size[1]; eye_display_w = eye_display_h 
        left_surf = pygame.surfarray.make_surface(np.transpose(left_rgb, (1, 0, 2)))
        left_surf = pygame.transform.smoothscale(left_surf, (eye_display_w, eye_display_h))
        right_surf = pygame.surfarray.make_surface(np.transpose(right_rgb, (1, 0, 2)))
        right_surf = pygame.transform.smoothscale(right_surf, (eye_display_w, eye_display_h))
        gap = int(grid_size[0] * 0.05); total_w = grid_size[0] + gap + eye_display_w + gap + eye_display_w
        font_size = 18; text_row_h = int(font_size * 1.8); padding = int(grid_size[0] * 0.03)
        window_w = total_w + padding * 2; window_h = grid_size[1] + text_row_h + padding * 2
        if self.window is None:
            pygame.init(); pygame.display.init()
            self.window = pygame.display.set_mode((window_w, window_h)); pygame.display.set_caption("Corner Maze")
        elif self.window.get_size() != (window_w, window_h): self.window = pygame.display.set_mode((window_w, window_h))
        if self.clock is None: self.clock = pygame.time.Clock()
        self.window.fill((255, 255, 255))
        self.window.blit(grid_surf, (padding, padding))
        eye_x_left = padding + grid_size[0] + gap; eye_x_right = eye_x_left + eye_display_w + gap
        self.window.blit(left_surf, (eye_x_left, padding)); self.window.blit(right_surf, (eye_x_right, padding))
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size); text_y = padding + grid_size[1] + 4
        session_text = f'{self.session_type}'
        font.render_to(self.window, (padding, text_y), session_text, size=font_size, fgcolor=(0, 0, 0))
        label_center_x = eye_x_left + (eye_display_w + gap + eye_display_w) // 2
        label_rect = font.get_rect(pose_label, size=font_size)
        font.render_to(self.window, (label_center_x - label_rect.width // 2, text_y), pose_label, size=font_size, fgcolor=(0, 0, 0))
        pygame.event.pump(); self.clock.tick(self.metadata["render_fps"]); pygame.display.flip()

    def plot_observation(self, observation_rgb):
        import matplotlib.pyplot as plt
        if not hasattr(self, 'fig'): self.fig, self.axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax in self.axes: ax.clear()
        self.axes[0].imshow(observation_rgb, interpolation='nearest'); self.axes[0].set_title("Full RGB"); self.axes[0].axis('off')
        channel_titles = ["Cues", "Walls", "Ledges"]
        for i in range(3):
            channel_img = observation_rgb[:, :, i]
            self.axes[i + 1].imshow(channel_img, cmap="gray", vmin=0, vmax=255, interpolation='nearest')
            self.axes[i + 1].set_title(channel_titles[i]); self.axes[i + 1].axis('off')
        plt.draw(); plt.pause(0.001)

    def _apply_action(self, action: int) -> str:
        if action == Actions.left:
            if self.agent_pos in CORNERS and self.last_pose in WELL_EXIT_POSES:
                new_pos_dir = CORNER_LEFT_TURN_WELL_EXIT.get(self.agent_pos)
                if new_pos_dir: self.agent_pos, self.agent_dir = new_pos_dir
                return "forward"
            else: self.agent_dir = (self.agent_dir - 1) % 4; return "turn"
        elif action == Actions.right: self.agent_dir = (self.agent_dir + 1) % 4; return "turn"
        elif action == Actions.forward:
            if self.agent_pose in WELL_EXIT_POSES:
                new_pos_dir = WELL_EXIT_FORWARD.get(self.agent_pos)
                if new_pos_dir: self.agent_pos, self.agent_dir = new_pos_dir
                return "forward"
            elif self.fwd_cell is None or self.fwd_cell.can_overlap(): self.agent_pos = tuple(self.fwd_pos); return "forward"
            else: return "wall_bump"
        elif action == ACTION_ENTER_WELL:
            new_pos_dir = WELL_ENTRY_PICKUP.get(self.agent_pos)
            if new_pos_dir: self.agent_pos, self.agent_dir = new_pos_dir
            return "forward"
        elif action == ACTION_PAUSE: return "pause"
        return "forward"

    def _update_state(self) -> None:
        self.cur_cell = self.grid.get(*self.agent_pos)
        self.last_pose = self.agent_pose
        self.agent_pose = (*self.agent_pos, self.agent_dir)
        state_type = self.grid_configuration_sequence[self.sequence_count][0]
        self.trajectory.append((*self.agent_pose, state_type))

    def _update_turn_scores(self) -> None:
        config = self.grid_configuration_sequence[self.sequence_count]
        if config[0] != STATE_TRIAL: return
        goal_locations = config[21:25]
        ns_goal_active = 1 in (goal_locations[0], goal_locations[2]); ew_goal_active = 1 in (goal_locations[1], goal_locations[3])
        if self.turn_score[0] is None:
            if ns_goal_active and self.agent_pos in TURN_ONE_SENW_MAP: self.turn_score[0] = TURN_ONE_SENW_MAP[self.agent_pos]
            elif ew_goal_active and self.agent_pos in TURN_ONE_NESW_MAP: self.turn_score[0] = TURN_ONE_NESW_MAP[self.agent_pos]
        elif self.turn_score[1] is None:
            if ns_goal_active:
                if self.agent_pos in TURN_TWO_SET_A: self.turn_score[1] = 1
                elif self.agent_pos in TURN_TWO_SET_B: self.turn_score[1] = 0
            elif ew_goal_active:
                if self.agent_pos in TURN_TWO_SET_A: self.turn_score[1] = 0
                elif self.agent_pos in TURN_TWO_SET_B: self.turn_score[1] = 1

    def _log_and_finalize_episode(self) -> None:
        self.episode += 1; episode_score = sum(self.episode_trial_scores) / len(self.episode_trial_scores)
        self.episode_scores.append(episode_score)
        self.episode_data_rows.append({'episode': self.episode, 'trial_scores': list(self.episode_trial_scores), 'turn_scores': [list(ts) for ts in self.episode_turn_scores], 'num_trials_completed': self.trial_count, 'total_steps': self.step_count, 'total_reward': self.session_reward, 'trajectory': list(self.trajectory)})

    def get_episode_data(self) -> pd.DataFrame: return pd.DataFrame(self.episode_data_rows)

    def _handle_exposure_well(self) -> tuple[str, bool]:
        well_idx = WELL_LOCATIONS.index(tuple(self.agent_pos))
        if well_idx in self.exposure_wells_remaining and self.exposure_iti_counter >= self.exposure_iti_threshold:
            self.exposure_reward_count += 1; self.exposure_wells_remaining.discard(well_idx); self.exposure_iti_counter = 0
            self.exposure_iti_threshold = int(random.gauss(EXPOSURE_ITI_STEPS, EXPOSURE_ITI_STD))
            if len(self.exposure_wells_remaining) == 0: self.exposure_wells_remaining = set(range(4))
            next_layout = self.maze_config_expa_lookup[frozenset(self.exposure_wells_remaining)]
            self.update_grid_configuration(next_layout)
            terminated = self.exposure_reward_count >= EXPOSURE_NUM_REWARDS
            return "well_reward", terminated
        else: return "well_empty", False

    def _handle_reward_well(self) -> tuple[str, bool]:
        self.trial_count += 1
        if self.session_num_trials == 1:
            if self.trial_score is None: self.trial_score = 1
            self.episode_trial_scores.append(self.trial_score); self.episode_turn_scores.append(list(self.turn_score))
            self._log_and_finalize_episode(); self.pseudo_session_score.appendleft(self.trial_score)
            return "well_reward", True
        elif self.session_num_trials > 1:
            if self.trial_score is None: self.trial_score = 1
            self.episode_trial_scores.append(self.trial_score); self.episode_turn_scores.append(list(self.turn_score))
            self.trial_score = None; self.turn_score = [None, None]; self.phase_step_count = 0
            if self.trial_count == self.session_num_trials: self._log_and_finalize_episode(); return "well_reward", True
            else: self._select_iti_configuration(); return "well_reward", False
        return "well_reward", False

    def _select_iti_configuration(self) -> None:
        self.sequence_count += 1; prev_config = self.grid_configuration_sequence[self.sequence_count - 1]
        goal_loc = next((i for i in range(4) if prev_config[21 + i] == 1), self.grid_configuration_sequence[self.sequence_count][-1])
        iti_configs = self.grid_configuration_sequence[self.sequence_count]
        start_arm_loc = next((arm for arm, idx in enumerate([2, 5, 8, 11]) if iti_configs[0][idx] == 0), self.grid_configuration_sequence[self.sequence_count + 1][-2])
        if start_arm_loc == goal_loc: config_idx, self.iti_type = 2, 0
        elif start_arm_loc == (goal_loc + 1) % 4: config_idx, self.iti_type = 1, 0
        else: config_idx, self.iti_type = 0, 1
        self._iti_config_idx = config_idx; self.update_grid_configuration(iti_configs[config_idx]); self.session_phase = STATE_ITI

    def _handle_trigger(self) -> None:
        self.trial_score = None; self.turn_score = [None, None]
        trigger_type = self.grid.get(*self.agent_pos).get_trigger_type()
        if trigger_type == 'A': self._iti_config_idx = 1; self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][1])
        elif trigger_type == 'B': self._iti_config_idx = 2; self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][2])
        elif trigger_type == 'S':
            self.sequence_count += 1; next_config = self.grid_configuration_sequence[self.sequence_count]; self.update_grid_configuration(next_config)
            if next_config[0] == STATE_PRETRIAL: self.session_phase = STATE_PRETRIAL; self.pretrial_step_count = 0
            else: self.session_phase = STATE_TRIAL
            self.phase_step_count = 0

    def _handle_pretrial(self) -> None:
        """Check pretrial position-based trigger with dynamic step requirements."""
        if self.session_phase != STATE_PRETRIAL:
            return
        
        self.pretrial_step_count += 1
        
        # DYNAMIC TRIGGER WINDOW using constant from constants.py
        required_steps = PRETRIAL_START_MIN_STEPS if self.trial_count == 0 else PRETRIAL_MIN_STEPS
        
        if self.pretrial_step_count < required_steps:
            return

        # Verify agent is at the dead-end trigger position for the active start arm
        if tuple(self.agent_pos) in PRETRIAL_TRIGGER_POSITIONS:
            self.sequence_count += 1
            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count])
            self.session_phase = STATE_TRIAL
            self.phase_step_count = 0
            self.pretrial_step_count = 0

    def _handle_exposure_b_phase_a(self) -> None:
        if self.session_phase != STATE_EXPB or self.expb_barrier_step is None: return
        self.expb_delay_counter += 1
        if self.expb_waiting_for_zone:
            if tuple(self.agent_pos) == self.expb_target_zone:
                self.expb_barrier_step += 1
                if self.expb_barrier_step >= len(EXPB_BARRIER_SEQUENCE):
                    self.session_phase = STATE_EXPA; self.update_grid_configuration(self.maze_config_expa_lookup[frozenset(range(4))])
                else:
                    seq = EXPB_BARRIER_SEQUENCE[self.expb_barrier_step]; self.update_grid_configuration(self.layouts[seq[0]])
                    self.expb_delay_counter = 0; self.expb_delay_target = seq[3]; self.expb_waiting_for_zone = False
        elif self.expb_delay_counter >= self.expb_delay_target:
            next_step = self.expb_barrier_step + 1
            if next_step >= len(EXPB_BARRIER_SEQUENCE):
                self.session_phase = STATE_EXPA; self.update_grid_configuration(self.maze_config_expa_lookup[frozenset(range(4))])
            else:
                seq = EXPB_BARRIER_SEQUENCE[next_step]
                if seq[1] == 'timed': self.expb_barrier_step = next_step; self.update_grid_configuration(self.layouts[seq[0]]); self.expb_delay_counter = 0; self.expb_delay_target = seq[3]
                elif seq[1] == 'zone': self.expb_waiting_for_zone = True; self.expb_target_zone = seq[2]

    def _handle_timeout(self) -> None:
        if self.session_num_trials == 1: self.trial_score = 0; self.episode_trial_scores.append(self.trial_score); self.episode_turn_scores.append(list(self.turn_score)); self._log_and_finalize_episode(); self.pseudo_session_score.appendleft(self.trial_score)
        elif self.session_num_trials > 1:
            if self.trial_score is None: self.trial_score = 0
            self.episode_trial_scores.append(self.trial_score); self.episode_turn_scores.append(list(self.turn_score)); self._log_and_finalize_episode()

    def _compute_reward(self, well_event: str | None, action: int) -> float:
        reward = STEP_TURN_COST if action in (0, 1) else STEP_FORWARD_COST
        if well_event == "well_reward": reward += WELL_REWARD_SCR
        return reward

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.step_count += 1; terminated = False; truncated = False
        self._apply_action(action); self._update_state(); self._update_turn_scores()
        well_event = None
        if isinstance(self.cur_cell, Well) and self.cur_cell.has_reward and self.session_phase == STATE_EXPA: well_event, t = self._handle_exposure_well(); terminated = terminated or t
        elif isinstance(self.cur_cell, Well) and self.cur_cell.has_reward: well_event, t = self._handle_reward_well(); terminated = terminated or t
        elif isinstance(self.cur_cell, Well) and not self.cur_cell.has_reward and self.last_pose[:2] not in WELL_LOCATIONS: self.trial_score = 0
        elif self.is_agent_on_obj(Trigger): self._handle_trigger()
        self._handle_pretrial(); self._handle_exposure_b_phase_a()
        if self.session_phase == STATE_EXPA: self.exposure_iti_counter += 1
        self.fwd_pos = self.front_pos; self.fwd_cell = self.grid.get(*self.fwd_pos); self.phase_step_count += 1
        reward = self._compute_reward(well_event, action); self.session_reward += reward
        if self.render_mode == "human": self.render()
        if self.step_count >= self.max_steps: self._handle_timeout(); truncated = True
        obs_mod = self._build_observation()
        info = {'agent_pos': self.agent_pos, 'terminated': terminated, 'truncated': truncated, 'episode_scores': self.episode_scores, 'session_reward': self.session_reward}
        return obs_mod, reward, terminated, truncated, info

    @staticmethod
    def _gen_mission(): return "corner maze mission"

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height); self.maze_state_array = [0] * 37
        self.put_obj_rect(Chasm(), 0, 0, width, height)
        for i in range(3):
            for j in range(9): self.grid.set(j+2, 4*i+2, None)
        for i in range(3):
            for j in range(3): self.grid.set(4*i+2, j+3, None); self.grid.set(4*i+2, j+7, None)
        for cx, cy in CUE_LOCATIONS: self.put_obj(Wall(color='cue_off_rgb'), cx, cy)
        for wx, wy in WELL_LOCATIONS: self.put_obj(Well(), wx, wy)
        self.grid_configuration_sequence, self.session_num_trials = self.gen_grid_configuration_sequence()
        self.update_grid_configuration(self.grid_configuration_sequence[0])
        self.agent_pos, self.agent_dir = self.gen_start_pose(); self.agent_pose = (*self.agent_pos, self.agent_dir); self.agent_start_pos = self.agent_pos
        if self.grid_configuration_sequence[0][0] == STATE_PRETRIAL: self.session_phase = STATE_PRETRIAL; self.pretrial_step_count = 0
        elif self.grid_configuration_sequence[0][0] == STATE_EXPA: self.session_phase = STATE_EXPA
        elif self.grid_configuration_sequence[0][0] == STATE_EXPB: self.session_phase = STATE_EXPB
        else: self.session_phase = STATE_TRIAL
        self.fwd_pos = self.front_pos; self.fwd_cell = self.grid.get(*self.fwd_pos); self.cur_cell = self.grid.get(*self.agent_pos)
        state_type = self.grid_configuration_sequence[0][0]; self.trajectory.append((*self.agent_pose, state_type))
        self.mission = "corner maze mission"