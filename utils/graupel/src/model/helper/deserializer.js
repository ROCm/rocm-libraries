import yaml from "js-yaml";
import { RR_SUBGRAPHS } from "../../constants.js";

async function fetchFileAsText(filePath) {
  return await fetch(filePath).then((res) => {
    return res.text();
  });
}

/**
 * Converts rocRoller to Cytoscape graph
 * @param {*} rr rocRoller yaml as JS object
 * @returns Cytoscape JS object
 */
export function convertRRtoCytoscape(rr) {
  /**
   * Creates a label (defaults to the most-specific type)
   * @param {Object} properties an object containing an element's properties
   * @returns a string label
   */
  function typeLabel(properties) {
    if (typeof properties.expressionStr !== "undefined") {
      return properties.expressionStr;
    }
    if (typeof properties.valueStr !== "undefined") {
      return properties.valueStr;
    }
    if (typeof properties.type !== "undefined") {
      const type = properties.type;
      return type.substring(type.lastIndexOf(".") + 1);
    }
    return undefined;
  }

  /**
   * Creates a list of classes
   * @param {string} str a period-separated string of classes
   * @returns array of classes (strings)
   */
  function typeClasses(prefix, str) {
    if (str === undefined || str === null) {
      return [];
    }
    const subtypes = str.split(".");
    let list = [subtypes[0].toLowerCase()];
    let prev = prefix + "-";
    for (let i = 1; i < subtypes.length; i++) {
      prev += subtypes[i].toLowerCase();
      list.push(prev);
      prev += "-";
    }
    return list;
  }

  /**
   * Converts a rocRoller element to a Cytoscape element
   * @param {string} prefix of subgraph
   * @param {string} id of element in the subgraph
   * @param {*} element
   * @returns Cytoscape element
   */
  function rrElementToNode(prefix, element) {
    const output = {
      group: "nodes",
      data: {
        id: prefix + element.id,
        label: typeLabel(element) + element.id,
        extra: element,
      },
      classes: [prefix, ...typeClasses(prefix, element.type)],
    };
    return output;
  }

  /**
   * Converts a rocRoller incident to a Cytoscape edge
   * @param {string} prefix of subgraph
   * @param {*} incident
   * @returns Cytoscape edge
   */
  function rrIncidentToEdge(prefix, incident) {
    return {
      group: "edges",
      data: {
        source: prefix + incident.src,
        target: prefix + incident.dst,
        extra: incident,
      },
      classes: [prefix],
    };
  }

  /**
   * Converts a rocRoller connection to a Cytoscape edge
   * @param {string} prefix of subgraph
   * @param {*} connection
   * @returns Cytoscape edge
   */
  function rrConnectionToEdge(prefix, connection) {
    return {
      group: "edges",
      data: {
        source: RR_SUBGRAPHS.CONTROL + connection.control,
        target: RR_SUBGRAPHS.COORDINATES + connection.coordinate,
        extra: connection,
      },
      classes: [prefix],
    };
  }

  /**
   * Converts connections to Cytoscape edges
   * @param {string} prefix of subgraph
   * @param {*} mapper rocRoller mapper structure
   * @returns object with edges
   */
  function convertMapper(prefix, mapper) {
    return {
      edges: mapper.connections.map((connection) =>
        rrConnectionToEdge(prefix, connection)
      ),
    };
  }

  /**
   * Converts hypergraph to Cytoscape elements object
   * @param {*} prefix of subgraph
   * @param {*} rr rocRoller hypergraph structure
   * @returns object with nodes and edges
   */
  function convertHypergraph(prefix, rr) {
    return {
      nodes: rr.elements.map((elem) => rrElementToNode(prefix, elem)),
      edges: rr.incidence.map((inci) => rrIncidentToEdge(prefix, inci)),
    };
  }

  const control = convertHypergraph(RR_SUBGRAPHS.CONTROL, rr.control);
  const coord = convertHypergraph(RR_SUBGRAPHS.COORDINATES, rr.coordinates);
  const mapper = convertMapper("mapper", rr.mapper);
  const output = {
    nodes: [...control.nodes, ...coord.nodes],
    edges: [...control.edges, ...coord.edges, ...mapper.edges],
  };

  return output;
}

/**
 * Converts a YAML file to a JS object
 * @param {string} inputPath relative path to input YAML file
 * @param {CallableFunction} conversionFunc conversion function, with rocRoller graph object as parameter and JS object as return
 */
export async function deserializeYamlToJS(inputPath, conversionFunc) {
  try {
    const raw = await fetchFileAsText(inputPath);
    const input = yaml.load(raw);
    const output = conversionFunc(input);
    return output;
  } catch (e) {
    console.error(e);
  }
}

/**
 * Converts a Assembly file to a JS object
 * @param {string} inputPath relative path to input Assembly file
 * @param {CallableFunction} conversionFunc conversion function, with rocRoller graph object as parameter and JS object as return
 */
export async function deserializeAsmToJS(inputPath, conversionFunc) {
  try {
    const text = await fetchFileAsText(inputPath);
    const startIdx = text.indexOf("---\n");
    const endIdx = text.indexOf("...\n");
    const input = yaml.load(text.substring(startIdx, endIdx));
    const graph = input["amdhsa.kernels"][0][".kernel_graph"];
    const output = conversionFunc(graph);
    return output;
  } catch (e) {
    console.error(e);
  }
}
