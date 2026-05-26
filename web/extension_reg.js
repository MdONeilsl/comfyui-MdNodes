import { app } from "../../../scripts/app.js";
//import { ComfyWidgets } from "../../../scripts/widgets.js";

import { set_node_widget_value } from "./libs/widgets_module.js";
import { SwitchController } from "./libs/SwitchController.js";
import { DynamicInputManager } from "./libs/DynamicInputManager.js";

app.registerExtension({
    name: "md.frontend_extension",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // Utility mdShowTensorShape
        if (nodeData.name === "mdShowTensorShape" || nodeData.name === "mdImageToBase64") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) { /// Need to be old function declaration for "this"
                onExecuted?.apply(this, arguments);
                const val = message?.info ? message.info.join("\n") : "No data received";
                set_node_widget_value(this, "info_display", val);
            };
        }

        if (nodeData.name === "mdSwitchControl") {
            // Store original prototype methods we need to wrap
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            const origOnConfigure = nodeType.prototype.onConfigure;
            const origOnDrawForeground = nodeType.prototype.onDrawForeground;
            const origOnMouseDown = nodeType.prototype.onMouseDown;
            const origComputeSize = nodeType.prototype.computeSize;

            // --- Wrap onNodeCreated ---
            nodeType.prototype.onNodeCreated = function () {
                origOnNodeCreated?.apply(this, arguments);
                // Create the controller and attach it to the node instance
                this._switchController = new SwitchController(this);
            };

            // --- Wrap onConfigure (loading workflow) ---
            nodeType.prototype.onConfigure = function (info) {
                origOnConfigure?.apply(this, arguments);
                if (this._switchController) {
                    // Reload from current properties (already restored by ComfyUI)
                    const savedSwitches = this.properties?.switches || [];
                    if (savedSwitches.length === 0) {
                        this._switchController.rebuildFromSwitches([{ value: false, label: "Switch_0" }]);
                    } else {
                        this._switchController.rebuildFromSwitches(savedSwitches);
                    }
                }
            };

            // --- Wrap onDrawForeground ---
            nodeType.prototype.onDrawForeground = function (ctx) {
                origOnDrawForeground?.apply(this, arguments);
                this._switchController?.onDrawForeground(ctx);
            };

            // --- Wrap onMouseDown ---
            nodeType.prototype.onMouseDown = function (e, pos) {
                // Let controller handle toggle/label clicks first
                if (this._switchController?.onMouseDown(e, pos)) {
                    return true;
                }
                // Otherwise fall back to original handler
                return origOnMouseDown?.apply(this, arguments) || false;
            };

            // --- Wrap computeSize ---
            nodeType.prototype.computeSize = function () {
                let size = origComputeSize ? origComputeSize.apply(this, arguments) : [LiteGraph.NODE_WIDTH, 60];
                if (this._switchController) {
                    size = this._switchController.computeSize(size);
                }
                return size;
            };
        }

        if (nodeData.name === "mdMergeControls") {
            // Store original methods
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;

            // Wrap onNodeCreated to instantiate the manager
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                this._dynamicManager = new DynamicInputManager(this, "input_");
                // Initial setup: ensure at least one empty slot if none exist
                if (this.inputs?.length === 0) {
                    this.addInput("input_0", "STRING");
                } else {
                    this._dynamicManager.manageDynamicInputs();
                }
            };

            // Wrap onConnectionsChange
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                onConnectionsChange?.apply(this, arguments);
                this._dynamicManager?.onConnectionsChange(type, index, connected, link_info);
            };

            // Hook into graph loading events to disable dynamic changes during load
            const origOnGraphConfigured = nodeType.prototype.onGraphConfigured;
            nodeType.prototype.onGraphConfigured = function () {
                origOnGraphConfigured?.apply(this, arguments);
                if (this._dynamicManager) {
                    this._dynamicManager.setLoading(false);
                    this._dynamicManager.manageDynamicInputs();
                }
            };

            const origOnGraphLoading = nodeType.prototype.onGraphLoading;
            nodeType.prototype.onGraphLoading = function () {
                origOnGraphLoading?.apply(this, arguments);
                if (this._dynamicManager) {
                    this._dynamicManager.setLoading(true);
                }
            };
        }

    },
});
