

export class DynamicInputManager {
    constructor(node, baseName = "input_") {
        this.node = node;
        this.baseName = baseName;
        this._loading = false;
    }

    // Call this when the graph starts loading (to avoid unwanted deletions)
    setLoading(loading) {
        this._loading = loading;
    }

    // Main logic: ensure there is always an empty slot and remove trailing unused ones
    manageDynamicInputs() {
        if (this._loading) return;

        const inputs = this.node.inputs || [];
        let hasEmptySlot = false;

        // 1. Check if there is an empty slot available
        for (let i = 0; i < inputs.length; i++) {
            if (!inputs[i].link) {
                hasEmptySlot = true;
                break;
            }
        }

        // 2. If all slots are filled, add a new one
        if (!hasEmptySlot) {
            const nextIndex = inputs.length;
            this.node.addInput(`${this.baseName}${nextIndex}`, "STRING");
        }

        // 3. Cleanup unused slots at the bottom, but ALWAYS keep at least the first one
        for (let i = inputs.length - 1; i > 0; i--) {
            if (!inputs[i].link && !inputs[i - 1].link) {
                this.node.removeInput(i);
            } else {
                break;
            }
        }

        this.node.setSize(this.node.computeSize());
    }

    // Optional: handle onConnectionsChange event
    onConnectionsChange(type, index, connected, link_info) {
        if (type === LiteGraph.INPUT) {
            this.manageDynamicInputs();
        }
    }
}
