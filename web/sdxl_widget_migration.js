import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FluxTrainer.SDXLWidgetMigration",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "InitSDXLLoRATraining") return;

        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (data) {
            const values = data?.widgets_values;
            // Older workflows stored gradient_dtype at index 11, before this checkbox was inserted.
            if (Array.isArray(values) && ["fp32", "fp16", "bf16"].includes(values[11])) {
                const migrated = [...values];
                migrated.splice(10, 0, false);
                data = { ...data, widgets_values: migrated };
            }
            return configure.call(this, data);
        };
    }
});
