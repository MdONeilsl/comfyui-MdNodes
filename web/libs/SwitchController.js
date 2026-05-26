// ----------------------------------------------------------------------
// Single class managing all dynamic switch behavior for one node instance
// ----------------------------------------------------------------------
export class SwitchController {
    constructor(node) {
        this.node = node;
        this._stateWidget = null;
        this.init();
    }

    // ---------- Initialization ----------
    init() {
        // Ensure properties exist
        if (!this.node.properties) this.node.properties = {};
        if (!this.node.properties.switches) this.node.properties.switches = [];

        // Runtime switch array (each: { value: boolean, label: string })
        this.node.switches = [];

        // Create hidden widget for JSON persistence (if not already present)
        if (!this.node.widgets || !this.node.widgets.find(w => w.name === "switches_state")) {
            this._stateWidget = this.node.addWidget("string", "switches_state", "", () => { }, { hidden: true });
            this._stateWidget.hidden = true;
        } else {
            this._stateWidget = this.node.widgets.find(w => w.name === "switches_state");
        }

        // "Add Switch" button
        this.node.addWidget("button", "➕ Add Switch", null, () => {
            this.addSwitch(false, `Switch_${this.node.switches.length}`);
        });

        // Initial load: either from saved properties or create a default switch
        if (this.node.properties.switches.length === 0) {
            // Node is fresh (Python default had 64 outputs) -> single default switch
            this.rebuildFromSwitches([{ value: false, label: "Switch_0" }]);
        } else {
            this.rebuildFromSwitches(this.node.properties.switches);
        }
    }

    // ---------- Output management ----------
    syncOutputs(targetCount) {
        const currentCount = this.node.outputs.length;

        // Rename existing outputs to invisible (non‑breaking space)
        for (let i = 0; i < currentCount; i++) {
            if (this.node.outputs[i]) {
                this.node.outputs[i].name = ".";
                this.node.outputs[i].label = "\u00A0";
            }
        }

        if (currentCount === targetCount) return;

        if (currentCount < targetCount) {
            for (let i = currentCount; i < targetCount; i++) {
                this.node.addOutput(".", "STRING");
            }
        } else {
            for (let i = currentCount - 1; i >= targetCount; i--) {
                this.node.removeOutput(i);
            }
        }
        this.node.setSize(this.node.computeSize());
    }

    // ---------- State synchronization ----------
    updateHiddenWidget() {
        if (this._stateWidget) {
            this._stateWidget.value = JSON.stringify(this.node.switches);
        }
    }

    syncProperties() {
        this.node.properties.switches = this.node.switches.map(sw => ({
            value: !!sw.value,
            label: sw.label || ""
        }));
        this.updateHiddenWidget();
    }

    // ---------- Core rebuild logic ----------
    rebuildFromSwitches(switchesArray) {
        // Adjust output count (preserves existing connections for retained slots)
        this.syncOutputs(switchesArray.length);

        // Update internal switch data
        this.node.switches = switchesArray.map((sw, idx) => ({
            value: !!sw.value,
            label: sw.label || `Switch_${idx}`
        }));

        this.syncProperties();
        this.node.setDirtyCanvas(true, true);
        app.graph.setDirtyCanvas(true);
    }

    // ---------- User actions ----------
    addSwitch(value, label) {
        const newSwitch = {
            value: !!value,
            label: label || `Switch_${this.node.switches.length}`
        };
        const newSwitches = [...this.node.switches, newSwitch];
        this.rebuildFromSwitches(newSwitches);
    }

    // ---------- UI Drawing (called from node.onDrawForeground) ----------
    onDrawForeground(ctx) {
        // ---- NEW: Skip drawing when node is minimized (collapsed) ----
        if (this.node.collapsed) return;

        const slotHeight = LiteGraph.NODE_SLOT_HEIGHT || 20;
        const startY = this.node.widgets_start_y || LiteGraph.NODE_TITLE_HEIGHT;

        for (let i = 0; i < this.node.switches.length; i++) {
            const y = startY + (i * slotHeight) - (slotHeight * 1.25);
            const sw = this.node.switches[i];

            // Label
            ctx.fillStyle = "#DDD";
            ctx.font = "12px Arial";
            ctx.textAlign = "left";
            ctx.fillText(sw.label, 10, y + 14);

            // Toggle switch background & knob
            const width = 50;
            const height = 16;
            const x = this.node.size[0] - width - 20;

            ctx.fillStyle = sw.value ? "#4CAF50" : "#666";
            ctx.beginPath();
            ctx.roundRect(x, y + 2, width, height, 8);
            ctx.fill();

            ctx.fillStyle = "#FFF";
            const knobX = sw.value ? x + width - height : x;
            ctx.beginPath();
            ctx.roundRect(knobX, y + 2, height, height, 8);
            ctx.fill();
        }
    }

    // ---------- Mouse interaction (called from node.onMouseDown) ----------
    onMouseDown(e, pos) {
        // ---- NEW: Ignore clicks when node is minimized (collapsed) ----
        if (this.node.collapsed) return false;

        const slotHeight = LiteGraph.NODE_SLOT_HEIGHT || 20;
        const startY = this.node.widgets_start_y || LiteGraph.NODE_TITLE_HEIGHT;

        for (let i = 0; i < this.node.switches.length; i++) {
            const y = startY + (i * slotHeight) - (slotHeight * 1.25);
            const toggleWidth = 50;
            const toggleHeight = 16;
            const toggleX = this.node.size[0] - toggleWidth - 20;
            const labelX = 10;
            const labelWidth = toggleX - 20;

            // Click on toggle switch
            if (pos[0] >= toggleX && pos[0] <= toggleX + toggleWidth &&
                pos[1] >= y && pos[1] <= y + toggleHeight) {
                this.node.switches[i].value = !this.node.switches[i].value;
                this.syncProperties();
                this.node.setDirtyCanvas(true, true);
                app.graph.setDirtyCanvas(true);
                return true; // event handled
            }

            // Click on label (rename)
            if (pos[0] >= labelX && pos[0] <= labelX + labelWidth &&
                pos[1] >= y && pos[1] <= y + toggleHeight) {
                app.canvas.prompt("Name:", this.node.switches[i].label, (newLabel) => {
                    if (newLabel !== null && newLabel.trim() !== "") {
                        this.node.switches[i].label = newLabel;
                        this.syncProperties();
                        this.node.setDirtyCanvas(true, true);
                        app.graph.setDirtyCanvas(true);
                    }
                }, e);
                return true;
            }
        }
        return false; // not handled, let original handler run
    }

    // ---------- Dynamic node sizing ----------
    computeSize(baseSize) {
        // ---- NEW: When collapsed, use only the base size (title bar) ----
        if (this.node.collapsed) return baseSize;

        const slotHeight = LiteGraph.NODE_SLOT_HEIGHT || 20;
        const minHeight = (this.node.widgets_start_y || LiteGraph.NODE_TITLE_HEIGHT) +
            (this.node.outputs?.length || 0) * slotHeight;
        if (baseSize[1] < minHeight) baseSize[1] = minHeight;
        return baseSize;
    }
}