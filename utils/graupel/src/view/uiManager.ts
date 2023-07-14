import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";
import dagre from "cytoscape-dagre";
import elk from "cytoscape-elk";
import cola from "cytoscape-cola";
import svg from "cytoscape-svg";
import cxtmenu from "cytoscape-cxtmenu";

import * as Manipulator from "../controller/manipulator";

import Buttons from "./components/buttons";
import Cli from "./components/cli";
import ClickEventListeners from "./helper/clickListeners";
import RightClickContextMenu from "./helper/rightClickContextMenu";
import StringUtils from "../utility/stringUtils";

cytoscape.use(fcose);
cytoscape.use(dagre);
cytoscape.use(elk as any);
cytoscape.use(cola);
cytoscape.use(svg);
cytoscape.use(cxtmenu);

// all functions dependent on cy need to be refreshed on new object
// should be called as cy.ready(refreshUI)
export function attachGuiListeners(cy) {
  RightClickContextMenu.setup(cy);
  ClickEventListeners.setupOnClick(document.getElementById("info"), cy);
}

const searchBox = document.getElementById("form1") as any;
Cli.setup(searchBox);
Buttons.addClickableButton("Go", () => {
  Cli.getInputAndSearch(searchBox);
});

Buttons.addToggleButton(
  "Drag/select subgraphs",
  () => Manipulator.dragSubgraphs(true),
  () => Manipulator.dragSubgraphs(false)
);

[
  Manipulator.redoLayout,
  Manipulator.unhighlight,
  Manipulator.onlyShowHighlighted,
  function Help() {
    console.log(Manipulator.help());
  },
].forEach((func) => {
  Buttons.addClickableButton(StringUtils.beautifyCamelCase(func.name), func);
});
