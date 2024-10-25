import os
import sys
import lightning.pytorch as pl
import hydra
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402
# from ocr.utils import draw_boxes  # 시각화 함수

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'

@hydra.main(config_path=CONFIG_DIR, config_name='validate', version_base='1.2')
def validate_and_visualize(config):
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(config)

    trainer = pl.Trainer()

    ckpt_path = config.get("checkpoint_path")
    assert ckpt_path is not None, "checkpoint_path must be provided for validation"

    # Validation 진행
    trainer.validate(model=model_module, datamodule=data_module, ckpt_path=ckpt_path)

    # 시각화를 위해 저장된 validation 예측 값 사용
    for image_filename, pred_boxes in model_module.validation_step_outputs.items():
        image_path = os.path.join(config.dataset_dir, image_filename)  # 이미지 경로 설정
        gt_polys = data_module.dataset['val'].anns[image_filename]  # Ground Truth 값 불러오기
        
        # 예측 결과 시각화
        # image = draw_boxes(image_path, pred_boxes, gt_polys)
        image = cv2.polylines(image_path, [pred_boxes], gt_polys)
        output_path = f"./visualization/{image_filename}"  # 결과 이미지 저장 경로
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    validate_and_visualize()