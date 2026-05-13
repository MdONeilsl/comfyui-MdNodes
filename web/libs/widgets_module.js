

export const set_node_widget_value = (node, widget_id, value) => {
    let widget = node.widgets?.find(w => w.name === widget_id);
    if (!widget) return;
    widget.value = value;
};
