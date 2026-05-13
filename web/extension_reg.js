import { app } from "../../../scripts/app.js";
//import { ComfyWidgets } from "../../../scripts/widgets.js";

import { set_node_widget_value } from "./libs/widgets_module.js";

app.registerExtension({
    name: "md.frontend_extension",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // Utility mdShowTensorShape
        if (nodeData.name === "mdShowTensorShape" || nodeData.name ===  "mdImageToBase64") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) { /// Need to be old function declaration for "this"
                onExecuted?.apply(this, arguments);
                const val = message?.info ? message.info.join("\n") : "No data received";
                set_node_widget_value(this, "info_display", val);
            };
        }
    },
});
