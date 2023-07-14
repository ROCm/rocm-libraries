module Buttons {
  export function addClickableButton(
    id: string,
    onClick: Function,
    ...args: any[]
  ) {
    const buttons = document.querySelector("#buttons");
    const newButton = document.createElement("button");
    newButton.id = id;
    newButton.type = "button";
    newButton.classList.add("btn");
    newButton.classList.add("btn-secondary");
    newButton.addEventListener("click", function () {
      onClick(...args);
    });
    buttons!.appendChild(newButton);
    const text = document.createTextNode(id);
    newButton.appendChild(text);
  }

  export function addToggleButton(text, onPush, onRelease) {
    const buttons = document.querySelector("#buttons");
    // Add a new button to the #buttons div
    let newButton = document.createElement("button");
    newButton.textContent = text;
    newButton.type = "button";
    newButton.classList.add("btn");
    newButton.classList.add("btn-secondary");
    newButton.setAttribute("data-toggle", "button");
    buttons!.appendChild(newButton);

    // Set initial state and event listener
    let isPushed = false;

    newButton.addEventListener("click", function () {
      isPushed = !isPushed; // Toggle the state

      if (isPushed) {
        newButton.classList.add("pushed"); // Apply a CSS class for the pushed state
        onPush();
      } else {
        newButton.classList.remove("pushed"); // Remove the CSS class when unpushed
        onRelease();
      }
    });
  }
}

export default Buttons;
