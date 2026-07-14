"""Inspect UDCP transmission and background-light feature maps."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as transform

from uwir.cli.train import _add_physics_channels, _resolve_physics_extractor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--prior-method", default="udcp", choices=("udcp", "gdcp", "gupdm"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    image = Image.open(args.image).convert("RGB").resize((256, 256))
    features = _add_physics_channels(
        transform.to_tensor(image),
        "tb",
        _resolve_physics_extractor(args.prior_method),
    )
    transmission, background = features[3].numpy(), features[4].numpy()
    print("transmission:", transmission.min(), transmission.max(), transmission.mean())
    print("background:", background.min(), background.max(), background.mean())

    if args.show:
        _, axes = plt.subplots(1, 2)
        axes[0].imshow(transmission, cmap="gray")
        axes[0].set_title("Transmission")
        axes[1].imshow(background, cmap="gray")
        axes[1].set_title("Background light")
        plt.show()


if __name__ == "__main__":
    main()
