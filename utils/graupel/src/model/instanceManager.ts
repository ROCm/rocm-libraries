import * as Manipulator from "../controller/manipulator";
import { attachGuiListeners as attachGuiListeners } from "../view/uiManager";

import cytoscape from "cytoscape";
import * as Deserializer from "./helper/deserializer";
import * as StylesheetGenerator from "./helper/stylesheetGenerator";
import StringUtils from "../utility/stringUtils";

import { Instance } from "./instanceManger";

/**
 * Creates element remove/restore functions ensuring no elements are permanantly removed
 * @param cy
 * @returns [remove(selector), restore(selector)]
 */
function createRemoveRestoreAPI(cy) {
  let removed = cy.collection();
  return [
    (selector: any = undefined) => {
      const selected = cy.$(selector).remove();
      removed = removed.union(selected);
      return selected;
    },
    (selector: any = undefined) => {
      let selected = cy.collection();
      try {
        selected = removed.filter(selector).restore();
      } catch (e) {
        // edges can only be restored if their source and target exist
        // restore as many edges as possible
        removed.filter(selector).forEach((elem) => {
          try {
            elem.restore();
            selected = selected.union(elem);
          } catch (e) { }
        });
      }
      removed = removed.difference(selected);
      return selected;
    },
  ];
}

let instances: Record<string, Instance> = {};

/**
 * Creates new instance
 * @param id of instance
 * @param path to file
 * @returns instance
 */
function _newInstance(id: string, path: string) {
  const [cy, style] = newCytoscape(path);
  const [removeElements, restoreElements] = createRemoveRestoreAPI(cy);
  const newMounted = {
    id: id,
    cy: cy,
    removeElements: removeElements,
    restoreElements: restoreElements,
    style: style,
  };
  cy.ready(() => attachGuiListeners(cy));
  instances[id] = newMounted;
  return newMounted;
}

/**
 * Unmounts previous instance and creates new instance
 * Mutates current instance
 * @param id of instance
 * @param path to file
 * @returns instance
 */
function newInstance(id: string = "", path: string) {
  if (id === "") {
    (newInstance as any).defaultIncrementalId =
      (newInstance as any).defaultIncrementalId || 0;
    id = ((newInstance as any).defaultIncrementalId++).toString();
  }
  CurrentInstance.cy.unmount();
  CurrentInstance = _newInstance(id, path);
  cy = CurrentInstance.cy;
  return CurrentInstance;
}

export let CurrentInstance: Instance = _newInstance("sample", "./kernels/sample.s");
export let cy = CurrentInstance.cy;

/**
 * Creates a new Cytoscape object from a file
 * @param path to file
 * @returns [cytoscape instance, stylesheet]
 */
function newCytoscape(path: string) {
  const keyedGroupFormat = loadFile(path);
  const rawJson = {
    container: document.getElementById("cy"),
    elements: keyedGroupFormat,
    style: keyedGroupFormat.then((res) =>
      StylesheetGenerator.generateStylesheet(res)
    ) as any,
  };
  const cy = cytoscape(rawJson as any);
  cy.ready(() => {
    Manipulator.fix();
    Manipulator.dragSubgraphs(false);
  });
  return [cy, rawJson.style];
}

/**
 * @returns current instances
 */
function listInstances() {
  return instances;
}

/**
 * Restores pre-existing instance
 * Mutates current instance
 * @param id of instance to be loaded
 * @returns instance object
 */
function restoreInstance(id: string) {
  CurrentInstance.cy.unmount();
  CurrentInstance = instances[id];
  cy = CurrentInstance.cy;
  CurrentInstance.cy.mount(document.getElementById("cy"));
  CurrentInstance.cy.ready(() => {
    CurrentInstance.style.then((res) => {
      CurrentInstance.cy.style(res);
    });
  });
  cy.ready(() => attachGuiListeners(cy));
  return CurrentInstance;
}

/**
 * Loads Cytoscape elements object from file
 * @param path to file
 * @returns promise resolving to Cytoscape elements object
 */
function loadFile(path: string): Promise<Object> {
  // TODO: move to deserializer
  let keyedGroupFormat = new Promise(() => undefined);
  try {
    switch (StringUtils.getFileExtension(path)) {
      case "s":
      case "asm":
      case "S":
        keyedGroupFormat = Deserializer.deserializeAsmToJS(
          path,
          Deserializer.convertRRtoCytoscape
        );
        break;
      case "yml":
      case "yaml":
        keyedGroupFormat = Deserializer.deserializeYamlToJS(
          path,
          Deserializer.convertRRtoCytoscape
        );
        break;
      default:
        keyedGroupFormat = new Promise(() => {
          throw new Error("Extension not compatible");
        });
    }
    return keyedGroupFormat;
  } catch (e) {
    console.log(e)
  }
}

/**
 * Replaces current instance with data from a file
 * Mutates current instance
 * @param path to file
 * @returns current instance
 */
function replaceInstance(path: string) {
  const result = Deserializer.deserializeAsmToJS(
    path,
    Deserializer.convertRRtoCytoscape
  );
  CurrentInstance.cy.elements().remove();
  result.then((keyedGroup) => {
    CurrentInstance.cy.json({
      style: StylesheetGenerator.generateStylesheet(keyedGroup),
    });
    CurrentInstance.cy.add(keyedGroup);
    cy = CurrentInstance.cy;
    cy.ready(() => {
      Manipulator.fix();
      Manipulator.dragSubgraphs(false);
    });
  });
  return cy;
}

/**
 * @param id
 * @returns deleted instance
 */
function deleteInstance(id: string) {
  return delete instances[id];
}

/**
 * Evaluates code in proper global context
 * Can mutate anything in global scope of this file
 * @param code to be ran
 * @returns result of running the code
 */
function doEval(code: string) {
  return eval(code);
}

export const Instances = {
  newInstance: newInstance,
  listInstances: listInstances,
  restoreInstance: restoreInstance,
  replaceInstance: replaceInstance,
  deleteInstance: deleteInstance,
  doEval: doEval,
};
