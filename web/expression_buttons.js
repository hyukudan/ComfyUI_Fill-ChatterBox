import { app } from "../../scripts/app.js";

// Paralinguistic expression tags supported by Turbo model
const EXPRESSION_TAGS = [
    { label: "😂", tag: "[laugh]", title: "Laugh" },
    { label: "😮‍💨", tag: "[sigh]", title: "Sigh" },
    { label: "😲", tag: "[gasp]", title: "Gasp" },
    { label: "🤭", tag: "[chuckle]", title: "Chuckle" },
    { label: "😷", tag: "[cough]", title: "Cough" },
    { label: "🤧", tag: "[sniff]", title: "Sniff" },
    { label: "😩", tag: "[groan]", title: "Groan" },
    { label: "🤫", tag: "[shush]", title: "Shush" },
    { label: "😤", tag: "[clear throat]", title: "Clear Throat" },
];

app.registerExtension({
    name: "Fill-ChatterBox.expressionButtons",

    async nodeCreated(node) {
        // Only apply to Turbo TTS node
        if (node.comfyClass !== "FL_ChatterboxTurboTTS") {
            return;
        }

        // Find the text widget
        const textWidget = node.widgets?.find(w => w.name === "text");
        if (!textWidget) {
            return;
        }

        // Create container for expression buttons
        const buttonContainer = document.createElement("div");
        buttonContainer.style.cssText = `
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            padding: 8px 0;
            justify-content: center;
        `;

        // Create buttons for each expression
        EXPRESSION_TAGS.forEach(expr => {
            const btn = document.createElement("button");
            btn.textContent = expr.label;
            btn.title = `Insert ${expr.title} ${expr.tag}`;
            btn.style.cssText = `
                padding: 4px 8px;
                font-size: 14px;
                border: 1px solid #666;
                border-radius: 4px;
                background: #2a2a2a;
                color: #fff;
                cursor: pointer;
                transition: all 0.2s;
                min-width: 32px;
            `;

            btn.addEventListener("mouseenter", () => {
                btn.style.background = "#4a4a4a";
                btn.style.borderColor = "#888";
            });

            btn.addEventListener("mouseleave", () => {
                btn.style.background = "#2a2a2a";
                btn.style.borderColor = "#666";
            });

            btn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();

                // Insert tag into text widget
                const currentText = textWidget.value || "";
                const inputEl = textWidget.inputEl;

                if (inputEl && document.activeElement === inputEl) {
                    // If textarea is focused, insert at cursor position
                    const start = inputEl.selectionStart;
                    const end = inputEl.selectionEnd;
                    const newText = currentText.substring(0, start) + expr.tag + " " + currentText.substring(end);
                    textWidget.value = newText;

                    // Update cursor position
                    const newPos = start + expr.tag.length + 1;
                    setTimeout(() => {
                        inputEl.setSelectionRange(newPos, newPos);
                        inputEl.focus();
                    }, 0);
                } else {
                    // If not focused, append to end
                    textWidget.value = currentText + (currentText.endsWith(" ") || currentText === "" ? "" : " ") + expr.tag + " ";
                }

                // Trigger change event
                if (textWidget.callback) {
                    textWidget.callback(textWidget.value);
                }
            });

            buttonContainer.appendChild(btn);
        });

        // Add custom widget for the buttons
        const buttonsWidget = node.addDOMWidget("expression_buttons", "div", buttonContainer, {
            serialize: false,
            hideOnZoom: false,
        });

        // Set widget height
        buttonsWidget.computedHeight = 50;

        // Move buttons widget after text widget
        const textIndex = node.widgets.indexOf(textWidget);
        const buttonsIndex = node.widgets.indexOf(buttonsWidget);
        if (buttonsIndex > textIndex + 1) {
            // Move to right after text widget
            node.widgets.splice(buttonsIndex, 1);
            node.widgets.splice(textIndex + 1, 0, buttonsWidget);
        }

        // Adjust node size
        node.setSize([node.size[0], node.size[1] + 60]);
    }
});
