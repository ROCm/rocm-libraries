import * as Manipulator from "../../controller/manipulator";
import StringUtils from "../../utility/stringUtils";

module RightClickContextMenu {
  export function setup(cy: any) {
    function getIdSelectorFromElement(elem) {
      return `#${elem.data().id}`;
    }
    const ctxmenuOptions = {
      commands: [
        Manipulator.highlight,
        Manipulator.unhighlight,
        Manipulator.highlightNeighbours,
        Manipulator.highlightOppositeNeighbours,
      ].map((func) => {
        return {
          content: StringUtils.beautifyCamelCase(func.name),
          contentStyle: {},
          select: (elem) => func(getIdSelectorFromElement(elem)),
          hover: (elem) => {},
          enabled: true,
        };
      }),
      outsideMenuCancel: 100,
    };
    const menu = cy.cxtmenu(ctxmenuOptions);
  }
}

export default RightClickContextMenu;
