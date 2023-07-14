import { CurrentInstance, cy, Instances } from "../model/instanceManager";
import * as Utils from "../model/helper/graphFixup";
import * as CmdParser from "./cliInterpreter";

import { saveAs } from "file-saver";
import { CY_CLASSES, CY_SELECTOR } from "../constants";
import StringUtils from "../utility/stringUtils";
import Help from "./specialManipulators/help";
import { Selector, Collection } from "cytoscape";

export function toggle(selector: Selector) {
  if (CurrentInstance.removeElements(selector).length === 0) {
    CurrentInstance.restoreElements(selector);
  }
}

export function remove(selector: Selector = CY_SELECTOR.SELECT_ALL) {
  CurrentInstance.removeElements(selector);
}

export function restore(selector: Selector = CY_SELECTOR.SELECT_ALL) {
  CurrentInstance.restoreElements(selector);
}

/**
 * Only show the highlighted elements
 * @param newLayout default is elk layout
 */
export function onlyShowHighlighted(newLayout: string | boolean = "elk") {
  CurrentInstance.removeElements(`.${CY_CLASSES.SUBGRAPH}`);
  CurrentInstance.removeElements(`.${CY_CLASSES.UNHIGHLIGHT}`);
  CurrentInstance.restoreElements(`.${CY_CLASSES.HIGHLIGHT}`);
  if (newLayout !== false && typeof newLayout !== "boolean") {
    CurrentInstance.cy.layout({ name: newLayout }).run();
  }
}

/**
 * Downloads current view as SVG (Scalable Vector Graphics) file
 */
export function download() {
  const blob = new Blob(
    [(CurrentInstance.cy as any).svg({ full: true, scale: 0.125 })],
    {
      type: "img/svg+xml",
    }
  );
  saveAs(blob, "graph.svg");
}

/**
 * Enables the movement and selection of the subgraphs (e.g. control, coordinate)
 * @param turnOn default is off
 */
export function dragSubgraphs(turnOn: Boolean | String | undefined | null) {
  if (turnOn) {
    (CurrentInstance.cy.nodes(`.${CY_CLASSES.SUBGRAPH}`) as any).unpanify();
    CurrentInstance.cy.nodes(`.${CY_CLASSES.SUBGRAPH}`).selectify();
  } else {
    (CurrentInstance.cy.nodes(`.${CY_CLASSES.SUBGRAPH}`) as any).panify();
    CurrentInstance.cy.nodes(`.${CY_CLASSES.SUBGRAPH}`).unselectify();
  }
}

/**
 * Padded fit all visible nodes
 * @param selector 
 * @returns 
 */
export function paddedFit(selector: string = CY_SELECTOR.SELECT_ALL) {
  return _paddedFit(CurrentInstance.cy.$(selector));
}

function _paddedFit(collection: Collection) {
  const PADDING_PIXELS = 250;
  return CurrentInstance.cy.fit(collection, PADDING_PIXELS);
}

export function select(selector, autoFit = true) {
  const result = CurrentInstance.cy.$(selector).select();
  if (autoFit) _paddedFit(result);
  return result;
}

export function highlight(selector: string = ":selected", autoFit = true) {
  const result = CurrentInstance.cy.$(selector).addClass(CY_CLASSES.HIGHLIGHT);
  if (autoFit) _paddedFit(result);
  return result;
}

export function idSelect(input: string) {
  return StringUtils.smartId(input, select);
}

export function idHighlight(input: string) {
  return StringUtils.smartId(input, highlight);
}

export function unhighlight(selector = CY_SELECTOR.SELECT_ALL) {
  CurrentInstance.cy
    .$(selector)
    .removeClass(CY_CLASSES.HIGHLIGHT)
    .union(CurrentInstance.cy.elements().removeClass(CY_CLASSES.UNHIGHLIGHT));
}

export function unselect(selector = ":selected") {
  return CurrentInstance.cy.$(selector).unselect();
}

/**
 * Highlights the neighbours of selected elements
 * @param selector default is highlighted elements
 * @returns 
 */
export function highlightNeighbours(selector = `.${CY_CLASSES.HIGHLIGHT}`) {
  const highlighted = CurrentInstance.cy
    .$(selector)
    .neighborhood()
    .difference(`.${CY_CLASSES.SUBGRAPH}`)
    .addClass(CY_CLASSES.HIGHLIGHT)
    .union(
      CurrentInstance.cy
        .$(selector)
        .children()
        .difference(`.${CY_CLASSES.SUBGRAPH}`)
        .addClass(CY_CLASSES.HIGHLIGHT)
    )
    .union(
      CurrentInstance.cy
        .$(selector)
        .parents()
        .difference(`.${CY_CLASSES.SUBGRAPH}`)
        .addClass(CY_CLASSES.HIGHLIGHT)
    );

  CurrentInstance.cy.elements().removeClass(CY_CLASSES.UNHIGHLIGHT);
  CurrentInstance.cy
    .elements()
    .filter((elem) => {
      return !elem.hasClass(CY_CLASSES.HIGHLIGHT);
    })
    .addClass(CY_CLASSES.UNHIGHLIGHT);

  CurrentInstance.cy.fit(
    highlighted
      .difference(`.${CY_CLASSES.SUBGRAPH}`)
      .difference(`.${CY_CLASSES.COMPOSITE}`)
  );
  return highlighted;
}

/**
 * Highlights elements in the opposite subgraph (e.g. control, coordinate) given the currently highlighted elements
 * 
 * Opposite here means the subgraph that does not include the most currently highlighted elements
 * 
 * e.g. if you have 4 control nodes and 3 coordinates nodes currently highlighted, coordinates nodes will be highlighted (as 4 > 3) with this function
 * 
 * @param removeMapper 
 * @returns 
 */
export function highlightOppositeNeighbours(removeMapper = true) {
  // TODO: cleanup
  CurrentInstance.restoreElements(`.${CY_CLASSES.MAPPER}`);
  const selectedCoord = CurrentInstance.cy
    .$(`.${CY_CLASSES.HIGHLIGHT}`)
    .filter(`.${CY_CLASSES.COORDINATES}`);
  const selectedControl = CurrentInstance.cy
    .$(`.${CY_CLASSES.HIGHLIGHT}`)
    .filter(`.${CY_CLASSES.CONTROL}`);

  let selectedClass = CY_CLASSES.CONTROL;
  let oppositeClass = CY_CLASSES.COORDINATES;
  if (selectedCoord.length > selectedControl.length) {
    [selectedClass, oppositeClass] = [oppositeClass, selectedClass];
  }

  CurrentInstance.cy
    .$(`.${CY_CLASSES.HIGHLIGHT}`)
    .neighborhood()
    .difference(`.${CY_CLASSES.SUBGRAPH}`)
    .difference("." + selectedClass)
    .addClass(CY_CLASSES.HIGHLIGHT);
  CurrentInstance.cy
    .$(`.${CY_CLASSES.HIGHLIGHT}`)
    .children()
    .difference(`.${CY_CLASSES.SUBGRAPH}`)
    .difference("." + selectedClass)
    .addClass(CY_CLASSES.HIGHLIGHT);
  CurrentInstance.cy
    .$(`.${CY_CLASSES.HIGHLIGHT}`)
    .parents()
    .difference(`.${CY_CLASSES.SUBGRAPH}`)
    .difference("." + selectedClass)
    .addClass(CY_CLASSES.HIGHLIGHT);

  CurrentInstance.cy.elements().removeClass(CY_CLASSES.UNHIGHLIGHT);
  CurrentInstance.cy
    .elements()
    .filter((elem) => {
      return !elem.hasClass(CY_CLASSES.HIGHLIGHT);
    })
    .addClass(CY_CLASSES.UNHIGHLIGHT);

  const PADDING_PIXELS = 10;
  if (removeMapper) {
    CurrentInstance.removeElements(CurrentInstance.cy.elements(`.${CY_CLASSES.MAPPER}`));
  }
  return CurrentInstance.cy.fit(
    CurrentInstance.cy
      .$(`.${CY_CLASSES.HIGHLIGHT}`)
      .difference(`.${CY_CLASSES.COMPOSITE}`),
    PADDING_PIXELS
  );
}

/**
 * Redoes layout following scott's algorithm
 * @param {removeMapper:true,restoreElements:true}
 * restoreElements: whether to restore all elements before applying layout
 * removeMapper: whether to remove mapper elements before applying layout
 */
export function redoLayout({
  removeMapper = true,
  restoreElements = true,
}: {
  removeMapper?: Boolean;
  restoreElements?: Boolean;
}) {
  if (removeMapper) {
    CurrentInstance.restoreElements(CY_SELECTOR.SELECT_ALL);
  }
  if (restoreElements) {
    CurrentInstance.removeElements(`.${CY_CLASSES.MAPPER}`);
  }
  Utils.redoLayout(CurrentInstance.cy);
}

export function highlightParents(selector = `.${CY_CLASSES.HIGHLIGHT}`) {
  CurrentInstance.cy.$(selector).parents().addClass(CY_CLASSES.HIGHLIGHT);
}

/**
 * Irreversibly simplifies graph for improved visual clarity
 * @param {groupOutgoers:false}
 * groupOutgoers: whether or not to group outgoing edges with similar destinations into a single edge and wraps the similar destinations into a composite
 */
export function fix({
  groupOutgoers = false,
}: { groupOutgoers?: Boolean } = {}) {
  CurrentInstance.removeElements(`.${CY_CLASSES.MAPPER}`);
  Utils.collapseCalmEdges(CurrentInstance.cy, `.${CY_CLASSES.CONTROL}`);
  Utils.expandNodeEdgeClasses(CurrentInstance.cy);
  try {
    Utils.createCompoundNodes(
      CurrentInstance.cy,
      CY_CLASSES.COORDINATES,
      "Coordinate Graph"
    );
    Utils.createCompoundNodes(CurrentInstance.cy, CY_CLASSES.CONTROL, "Control Graph");
  } catch (e) { }
  if (groupOutgoers) {
    console.log("grouping");
    Utils.groupOutgoers(CurrentInstance.cy, "", `${CY_CLASSES.CONTROL}-parent`, 6);
  }
  Utils.groupLinearSequences(CurrentInstance.cy, `.${CY_CLASSES.CONTROL}`);
  Utils.redoLayout(CurrentInstance.cy);
}
fix.aliases = ["f"] // for testing, currently unused

export function newInstance(path: string, id = "") {
  return Instances.newInstance(id, path);
}

export function listInstances() {
  return Instances.listInstances()
    + "use `restoreInstance <id>` to restore that instance";
}

export function restoreInstance(id: string) {
  return Instances.restoreInstance(id);
}

export function replaceInstance(path: string) {
  return Instances.replaceInstance(path);
}

export function deleteInstance(id: string) {
  return Instances.deleteInstance(id);
}

export function layout(name: string, selector: string = CY_SELECTOR.SELECT_ALL) {
  return CurrentInstance.cy.$(selector).layout({ name: name }).run();
}

/**
 * Console logs the help dialog
 * @param direction default "f" for forward; "b" for backwards
 * @returns help dialog
 */
export function help(direction: string = "f"): string {
  if (direction === "f") {
    return Help.f();
  }
  return Help.b();
}

/**
 * Returns manual for the state manipulator functions or the docs of a specific manipulator
 * @param name
 * @returns
 */
export function man(name: string): string {
  /**
   * Checks if any of the URLs provided exist, if so open it in a new tab. 
   */
  function checkAndOpen(urls: Array<string>) {
    const fetchPromises = urls.map(url =>
      fetch(url).then(res => {
        if (res.status === 200) {
          return url;
        }
      })
    );
    Promise.all(fetchPromises).then(resolvedUrls => {
      const filteredUrls = resolvedUrls.filter(url => url !== undefined);
      if (filteredUrls.length > 0) {
        window.open(filteredUrls[0]).focus();
      } else {
        console.log("No docs found, run `npm run docs` to generate them. ");
      }
    });
  }

  if (typeof name === "undefined") {
    const url = `${window.location.href}docs/modules/controller_manipulator.html`
    checkAndOpen([url]);
  }
  const [cyFunctions, userFunctions, aliasFunctions] =
    CmdParser.getAllFunctions();
  if (name in userFunctions) {
    checkAndOpen([`${window.location.href}docs/functions/controller_manipulator.${name}.html`,
    `${window.location.href}docs/functions/controller_manipulator.${name}-1.html`]);
  } else if (name in cyFunctions) {
    checkAndOpen([`https://js.cytoscape.org/#cy.${name}`])
  } else if (name in aliasFunctions) {
    return "This is an alias"; // TODO: link to function it calls when aliases attached to functions
  } else {
    return "No function found";
  }
}

export function run(code: string) {
  return Instances.doEval(code);
}
