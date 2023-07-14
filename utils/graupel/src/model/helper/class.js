/**
 * Gets the order of each class, that is the number of "-" in its string
 * TODO: less error-prone methods?
 * @param {String} classs
 * @returns {Number} order
 */
export function getOrderOfClass(classs) {
  return (classs.match(/-/g) || []).length;
}

/**
 * Gets the highest order class out of all the classes
 * @param {Array[String]} classes
 * @returns {String} class
 */
export function getHighestOrderClass(classes) {
  return classes.sort((a, b) => getOrderOfClass(b) - getOrderOfClass(a))[0];
}

/**
 * Filters for classes that have any of the provided prefixes
 * More formally, for classes C and prefixes P, return
 * {c ∈ C | ∃p ∈ P : p prefixes c}
 * @param {Array[String]} classes
 * @param {Iterable[String]} prefixes
 * @returns {Array[String]} filtered classes
 */
export function filterClassesByPrefix(classes, prefixes) {
  return classes.filter((classs) =>
    prefixes.some((prefix) => classs.includes(prefix))
  );
}
