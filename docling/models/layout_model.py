import copy
import logging
import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

from docling_core.types.doc import DocItemLabel
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image

from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import AcceleratorOptions
from docling.datamodel.settings import settings
from docling.models.base_model import BasePageModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder
from docling.utils.visualization import draw_clusters

_log = logging.getLogger(__name__)


class LayoutModel(BasePageModel):
    _model_repo_folder = "ds4sd--docling-models"
    _model_path = "model_artifacts/layout"

    TEXT_ELEM_LABELS = [
        DocItemLabel.TEXT,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.CAPTION,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.CODE,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.FORMULA,
    ]
    PAGE_HEADER_LABELS = [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]

    TABLE_LABELS = [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
    FIGURE_LABEL = DocItemLabel.PICTURE
    FORMULA_LABEL = DocItemLabel.FORMULA
    CONTAINER_LABELS = [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION]
    MODEL_TO_DOC_LABEL_MAP = {
        "title": "Title",
        "plain text": "Text",
        "figure": "Picture",
        "figure_caption": "Caption",
        "table": "Table",
        "table_caption": "Caption",
        "table_footnote": "Footnote",
        "isolate_formula": "Formula",
        "formula_caption": "Caption",
        # 'abandon': 'background' or skip
    }

    def __init__(
        self, artifacts_path: Optional[Path], accelerator_options: AcceleratorOptions
    ):
        device = decide_device(accelerator_options.device)

        if artifacts_path is None:
            artifacts_path = self.download_models() / self._model_path
        else:
            # will become the default in the future
            if (artifacts_path / self._model_repo_folder).exists():
                artifacts_path = (
                    artifacts_path / self._model_repo_folder / self._model_path
                )
            elif (artifacts_path / self._model_path).exists():
                warnings.warn(
                    "The usage of artifacts_path containing directly "
                    f"{self._model_path} is deprecated. Please point "
                    "the artifacts_path to the parent containing "
                    f"the {self._model_repo_folder} folder.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                artifacts_path = artifacts_path / self._model_path

        self.layout_predictor = LayoutPredictor(
            artifact_path=str(artifacts_path),
            device=device,
            num_threads=accelerator_options.num_threads,
        )

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        if not progress:
            disable_progress_bars()
        download_path = snapshot_download(
            repo_id="ds4sd/docling-models",
            force_download=force,
            local_dir=local_dir,
            revision="v2.1.0",
        )

        return Path(download_path)

    def draw_clusters_and_cells_side_by_side(
        self, conv_res, page, clusters, mode_prefix: str, show: bool = False
    ):
        """
        Draws a page image side by side with clusters filtered into two categories:
        - Left: Clusters excluding FORM, KEY_VALUE_REGION, and PICTURE.
        - Right: Clusters including FORM, KEY_VALUE_REGION, and PICTURE.
        Includes label names and confidence scores for each cluster.
        """
        scale_x = page.image.width / page.size.width
        scale_y = page.image.height / page.size.height

        # Filter clusters for left and right images
        exclude_labels = {
            DocItemLabel.FORM,
            DocItemLabel.KEY_VALUE_REGION,
            DocItemLabel.PICTURE,
        }
        left_clusters = [c for c in clusters if c.label not in exclude_labels]
        right_clusters = [c for c in clusters if c.label in exclude_labels]
        # Create a deep copy of the original image for both sides
        left_image = copy.deepcopy(page.image)
        right_image = copy.deepcopy(page.image)

        # Draw clusters on both images
        draw_clusters(left_image, left_clusters, scale_x, scale_y)
        draw_clusters(right_image, right_clusters, scale_x, scale_y)
        # Combine the images side by side
        combined_width = left_image.width * 2
        combined_height = left_image.height
        combined_image = Image.new("RGB", (combined_width, combined_height))
        combined_image.paste(left_image, (0, 0))
        combined_image.paste(right_image, (left_image.width, 0))
        if show:
            combined_image.show()
        else:
            out_path: Path = (
                Path(settings.debug.debug_output_path)
                / f"debug_{conv_res.input.file.stem}"
            )
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{mode_prefix}_layout_page_{page.page_no:05}.png"
            combined_image.save(str(out_file), format="png")

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        filepath = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        model = YOLOv10(filepath)
        print("GERMAN using model YOLOv10")

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "layout"):
                    assert page.size is not None
                    page_image = page.get_image(scale=1.0)
                    assert page_image is not None
                    page_image.save("./temp.png")

                    clusters = []
                    for ix, pred_item in enumerate(model.predict("./temp.png")):
                        print("GERMAN pred_item names", ix, pred_item.names)
                        print("GERMAN pred_item boxes", ix, pred_item.boxes)

                        # Get all box info
                        xyxy = pred_item.boxes.xyxy
                        confs = pred_item.boxes.conf
                        clses = pred_item.boxes.cls
                        names = pred_item.names

                        for j, box in enumerate(xyxy):
                            l, t, r, b = box.tolist()
                            conf = confs[j].item()
                            cls_idx = int(clses[j].item())

                            cls_idx = int(clses[j].item())
                            model_label = names.get(cls_idx, "unknown")

                            # Skip unwanted classes (e.g. 'abandon')
                            if model_label == "abandon":
                                continue

                            # Map to desired label
                            mapped_label = self.MODEL_TO_DOC_LABEL_MAP.get(
                                model_label, "unknown"
                            )
                            label = DocItemLabel(
                                mapped_label.lower().replace(" ", "_").replace("-", "_")
                            )
                            cluster = Cluster(
                                id=ix * j,
                                label=label,
                                confidence=conf,
                                bbox=BoundingBox.model_validate(
                                    {"l": l, "t": t, "r": r, "b": b}
                                ),
                                cells=[],
                            )

                            clusters.append(cluster)
                        # label = DocItemLabel(
                        #     f"label_pred_item_{ix}".format(ix=ix)
                        #     .lower()
                        #     .replace(" ", "_")
                        #     .replace("-", "_")
                        # )  # Temporary, until docling-ibm-model uses docling-core types
                        # cluster = Cluster(
                        #     id=ix,
                        #     label=label,
                        #     confidence=pred_item.boxes["conf"],
                        #     bbox=BoundingBox.model_validate(
                        #         {
                        #             "l": pred_item.boxes,
                        #             "r": pred_item,
                        #             "t": pred_item,
                        #             "b": pred_item,
                        #         }
                        #     ),
                        #     cells=[],
                        # )
                        # clusters.append(cluster)

                    if settings.debug.visualize_raw_layout:
                        self.draw_clusters_and_cells_side_by_side(
                            conv_res, page, clusters, mode_prefix="raw"
                        )

                    # Apply postprocessing

                    processed_clusters, processed_cells = LayoutPostprocessor(
                        page.cells, clusters, page.size
                    ).postprocess()
                    # processed_clusters, processed_cells = clusters, page.cells

                    page.cells = processed_cells
                    page.predictions.layout = LayoutPrediction(
                        clusters=processed_clusters
                    )

                if settings.debug.visualize_layout:
                    self.draw_clusters_and_cells_side_by_side(
                        conv_res, page, processed_clusters, mode_prefix="postprocessed"
                    )

                yield page
