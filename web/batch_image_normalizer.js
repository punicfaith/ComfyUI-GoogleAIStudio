// const { app } = window.comfyAPI.app;
// const { applyTextReplacements } = window.comfyAPI.utils;

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// import { app } from "../../../scripts/app.js";
// import { api } from '../../../scripts/api.js'
// import { ComfyWidgets } from "../../../scripts/widgets.js"

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

                return r;
            };
        }
    }
});
