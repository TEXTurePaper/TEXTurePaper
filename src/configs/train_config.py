from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger


@dataclass
class RenderConfig:
    """ Parameters for the Mesh Renderer """
    # Grid size for rendering during painting
    train_grid_size: int = 1200
    # Grid size of evaluation
    eval_grid_size: int = 1024
    # training camera radius range
    radius: float = 1.5
    # Set [0,overhead_range] as the overhead region
    overhead_range: float = 40
    # Define the front angle region
    front_range: float = 70
    # The front offset, use to rotate shape from code
    front_offset: float = 0.0
    # Number of views to use
    n_views: int = 8
    # Theta value for rendering during training
    base_theta: float = 60
    # Additional views to use before rotating around shape
    views_before: List[Tuple[float, float]] = field(default_factory=list)
    # Additional views to use after rotating around shape
    views_after: List[Tuple[float, float]] = field(default_factory=[[180, 30], [180, 150]].copy)
    # Whether to alternate between the rotating views from the different sides
    alternate_views: bool = True


@dataclass
class GuideConfig:
    """ Parameters defining the guidance """
    # Guiding text prompt
    text: str
    # The mesh to paint
    shape_path: str = 'shapes/spot_triangulated.obj'
    # Append direction to text prompts
    append_direction: bool = True
    # A Textual-Inversion concept to use
    concept_name: Optional[str] = None
    # Path to the TI embedding
    concept_path: Optional[Path] = None
    # A huggingface diffusion model to use
    diffusion_name: str = 'stabilityai/stable-diffusion-2-depth'
    # Scale of mesh in 1x1x1 cube
    shape_scale: float = 0.6
    # height of mesh
    dy: float = 0.25
    # texture image resolution
    texture_resolution: int = 1024
    # texture mapping interpolation mode from texture image, options: 'nearest', 'bilinear', 'bicubic'
    texture_interpolation_mode: str = 'bilinear'
    # Guidance scale for score distillation
    guidance_scale: float = 7.5
    # Use inpainting in relevant iterations
    use_inpainting: bool = True
    # The texture before editing
    reference_texture: Optional[Path] = None
    # The edited texture
    initial_texture: Optional[Path] = None
    # Whether to use background color or image
    use_background_color: bool = False
    # Background image to use
    background_img: str = 'textures/brick_wall.png'
    # Threshold for defining refine regions
    z_update_thr: float = 0.2
    # Some more strict masking for projecting back
    strict_projection: bool = True


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Seed for experiment
    seed: int = 0
    # Learning rate for projection
    lr: float = 1e-2
    # For Diffusion model
    min_timestep: float = 0.02
    # For Diffusion model
    max_timestep: float = 0.98
    # For Diffusion model
    no_noise: bool = False


@dataclass
class LogConfig:
    """ Parameters for logging and saving """
    # Experiment name
    exp_name: str
    # Experiment output dir
    exp_root: Path = Path('experiments/')
    # Run only test
    eval_only: bool = False
    # Number of angles to sample for eval during training
    eval_size: int = 10
    # Number of angles to sample for eval after training
    full_eval_size: int = 100
    # Export a mesh
    save_mesh: bool = True
    # Whether to show intermediate diffusion visualizations
    vis_diffusion_steps: bool = False
    # Whether to log intermediate images
    log_images: bool = True

    @property
    def exp_dir(self) -> Path:
        return self.exp_root / self.exp_name


@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)
