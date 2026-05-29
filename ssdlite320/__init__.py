__all__ = [
	"DefaultBoxes",
	"Encoder",
	"dboxes320_coco",
	"MOBILE_NET_BACKBONES",
	"Loss",
	"MobileNet",
	"SSD320",
]


def __getattr__(name):
	if name in {"DefaultBoxes", "Encoder", "dboxes320_coco"}:
		from .encoder import DefaultBoxes, Encoder, dboxes320_coco

		exports = {
			"DefaultBoxes": DefaultBoxes,
			"Encoder": Encoder,
			"dboxes320_coco": dboxes320_coco,
		}
		return exports[name]

	if name in {"MOBILE_NET_BACKBONES", "Loss", "MobileNet", "SSD320"}:
		from .model import MOBILE_NET_BACKBONES, Loss, MobileNet, SSD320

		exports = {
			"MOBILE_NET_BACKBONES": MOBILE_NET_BACKBONES,
			"Loss": Loss,
			"MobileNet": MobileNet,
			"SSD320": SSD320,
		}
		return exports[name]

	raise AttributeError(f"module 'ssdlite320' has no attribute {name!r}")
