import { CY_CLASSES } from "../../constants";

module ClickEventListeners {
  export function setupOnClick(infobox: HTMLElement | null, cy: any) {
    cy.on("click", "*", function (evt) {
      console.log(evt.target);
      (infobox as any).value = JSON.stringify({
        data: evt.target.data(),
        classes: evt.target.classes(),
      });
    });

    cy.on("dbltap", "*", function (evt) {
      evt.target.addClass(CY_CLASSES.HIGHLIGHT);
    });
  }
}

export default ClickEventListeners;
