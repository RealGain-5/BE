import sys
import json
import os
import torch

from model_loader import load_trained_model
from infer_resnet_None import (
    make_orbit_pils_sec9_from_bin,
    predict_rcp_single,
    make_temporal_orbit_pils,  # for Lazy Loading
    generate_gradcam_images,
)
from utils import image_to_base64

# cold start
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    SCRIPT_DIR, "model", "resnet18_orbit_v3_None.pth"
)

print(f"{MODEL_PATH}", file=sys.stderr)

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_trained_model(MODEL_PATH)
    model.to(device)
    print("[Python]model loaded successfully(cold start). waiting...", file=sys.stderr)
except Exception as e:
    print(f"error loading model: {e}", file=sys.stderr)
    sys.exit(1)


# Daemon loop
def main():
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break  # EOF (when parent process terminated...)

            # parsing command
            req = json.loads(line)
            command = req.get("command")
            payload = req.get("payload", {})
            bin_path = payload.get("bin_path")

            print(f"[Daemon] Received command: {command}", file=sys.stderr)
            print(f"[Daemon] Payload: {payload}", file=sys.stderr)

            response = {"status": "error", "data": None}

            # quick analyze => infer + overlay
            if command == "analyze":
                # generate image -> memory
                rcp_to_pil = make_orbit_pils_sec9_from_bin(bin_path)

                results = {}
                images_b64 = {}

                # infer and transform to Base64
                for rcp, pil_img in rcp_to_pil.items():
                    # infer
                    pred_class, prob = predict_rcp_single(model, class_names, pil_img)
                    results[rcp] = {
                        "prediction": pred_class,
                        "probabilities": {
                            name: float(p) for name, p in zip(class_names, prob)
                        },
                    }

                    # transform
                    gradcam_imgs = generate_gradcam_images(model, class_names, pil_img)

                    images_b64[rcp] = {
                        "orbit": image_to_base64(pil_img),  # original orbit
                        "heatmap": image_to_base64(
                            gradcam_imgs["heatmap"]
                        ),  # grad-cam heatmap
                        "overlay": image_to_base64(
                            gradcam_imgs["overlay"]
                        ),  # grad-cam overlay
                    }

                # final labeling
                final_label = (
                    "abnormal"
                    if any(r["prediction"] == "abnormal" for r in results.values())
                    else "normal"
                )

                response = {
                    "status": "ok",
                    "type": "anlysis_result",
                    "data": {
                        "final_label": final_label,
                        "results": results,
                        "images": images_b64,
                    },
                }

            elif command == "timeline":
                # heavy work
                # do only if user want => not always
                rcp_to_temporal = make_temporal_orbit_pils(bin_path)

                timeline_b64 = {}
                for rcp, img_list in rcp_to_temporal.items():
                    timeline_b64[rcp] = [image_to_base64(img) for img in img_list]

                response = {
                    "status": "ok",
                    "type": "timeline_result",
                    "data": timeline_b64,
                }

            else:
                response["message"] = f"Unknown command: {command}"

            # send response
            print(json.dumps(response))
            sys.stdout.flush()

        except Exception as e:
            # do not let process die
            print(f"[Daemon] ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

            err_reponse = {"status": "error", "message": str(e)}
            print(json.dumps(err_reponse))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
