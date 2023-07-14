import * as CmdParser from "../../controller/cliInterpreter";

module Cli {
  export function getInputAndSearch(textbox: HTMLElement) {
    const input = (textbox as any).value;
    const output = CmdParser.parse(input);
    if (typeof output !== "undefined") {
      console.log(output);
    }
  }

  export function setup(textbox: HTMLElement) {
    document.addEventListener("keypress", (event) => {
      if (event.ctrlKey && event.code === "Space") {
        event.preventDefault();
        event.stopPropagation();
        textbox.focus();
        (textbox as any).select();
      }
    });
    document.addEventListener("keypress", (event) => {
      if (event.code === "Enter" || event.code === "NumpadEnter") {
        event.preventDefault();
        event.stopPropagation();
        getInputAndSearch(textbox);
        (textbox as any).select();
      }
    });
  }
}

export default Cli;
