import { Core, Stylesheet } from "cytoscape";

type Instance = {
  cy: Core;
  id: String;
  removeElements: Function;
  restoreElements: Function;
  style: Promise<Stylesheet>;
};