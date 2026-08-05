// const { app } = window.comfyAPI.app;
// const { applyTextReplacements } = window.comfyAPI.utils;

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// import { app } from "../../../scripts/app.js";
// import { api } from '../../../scripts/api.js'
// import { ComfyWidgets } from "../../../scripts/widgets.js"

// canvas_shape 가 square 면 height 위젯을 감춘다. 두 노드가 같은 위젯을 쓰므로
// 여기 한 곳에 둔다 — 배치 입력 노드에는 "Update inputs" 만 없다.
function bindCanvasShape(node) {
    const shapeW = node.widgets?.find(w => w.name === "canvas_shape");
    const heightW = node.widgets?.find(w => w.name === "resolution_height");
    const widthW = node.widgets?.find(w => w.name === "resolution_value");
    if (!shapeW || !heightW) return;
    heightW._origType = heightW.type;
    heightW._origComputeSize = heightW.computeSize;
    const applyShape = () => {
        const rect = shapeW.value === "rectangle";
        heightW.type = rect ? heightW._origType : "hidden";
        heightW.computeSize = rect ? heightW._origComputeSize : () => [0, -4];
        if (widthW) widthW.label = rect ? "resolution_value (width)" : "resolution_value";
        node.setDirtyCanvas(true, true);
    };
    const prev = shapeW.callback;
    shapeW.callback = function (v) {
        const out = prev?.apply(this, arguments);
        applyShape();
        return out;
    };
    // 저장된 워크플로를 열 때도 맞춰 준다 (위젯 값이 실린 뒤에)
    setTimeout(applyShape, 0);
}

app.registerExtension({
    name: "ComfyUI.GoogleAIStudio.BatchImageNormalizerBatch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "BatchImageNormalizerBatch") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            bindCanvasShape(this);
            return r;
        };
    },
});

app.registerExtension({
    name: "ComfyUI.GoogleAIStudio.BatchImageNormalizer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // console.log("nodeData.name ", nodeData.name);
        if (nodeData.name === "BatchImageNormalizer") {
            console.log("BatchImageNormalizer registered");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this._type = "IMAGE";

                console.log("BatchImageNormalizer onNodeCreated", r);

                // Add the update button widget
                this.addWidget("button", "Update inputs", null, () => {
                    if (!this.inputs) {
                        this.inputs = [];
                    }

                    const inputcountWidget = this.widgets.find(w => w.name === "inputcount");
                    if (!inputcountWidget) {
                        console.error("inputcount widget not found");
                        return;
                    }

                    const target_number_of_inputs = inputcountWidget.value;
                    const num_inputs = this.inputs.filter(input => input.type === this._type).length;

                    if (target_number_of_inputs === num_inputs) return; // already set, do nothing

                    if (target_number_of_inputs < num_inputs) {
                        const inputs_to_remove = num_inputs - target_number_of_inputs;
                        for (let i = 0; i < inputs_to_remove; i++) {
                            this.removeInput(this.inputs.length - 1);
                        }
                    } else {
                        for (let i = num_inputs + 1; i <= target_number_of_inputs; ++i) {
                            this.addInput(`image_${i}`, this._type, { shape: 7 });
                        }
                    }
                });

                bindCanvasShape(this);

                return r;
            };
        }
    }
});
