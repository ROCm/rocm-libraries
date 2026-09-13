local review_pages = {
  ["architecture-component-model"] = "architecture-component-model.html",
  ["broker"] = "broker.html",
  ["cohort"] = "cohort.html",
  ["facade"] = "facade.html",
  ["hipblaslt-horizontal"] = "hipblaslt-horizontal.html",
  ["hipblaslt-facade-path"] = "hipblaslt-facade-path.html",
  ["manifest"] = "manifest.html",
  ["provider-adapter"] = "provider-adapter.html",
  ["provider-binding"] = "provider-binding.html",
  ["provider-module"] = "provider-module.html",
  ["provider-protocol"] = "provider-protocol.html",
  ["provider"] = "provider.html",
}

local first_h1_removed = false

local function remove_markdown_suffix(text)
  return text:gsub("([%w_%-]+)%.md", "%1")
end

function Link(element)
  local path, anchor = element.target:match("^([^#]+)(#.*)$")
  path = path or element.target
  anchor = anchor or ""
  local name = path:match("([^/]+)%.md$")

  if name and review_pages[name] then
    element.target = review_pages[name] .. anchor
    return element
  end

  if path:match("%.md$") then
    return element.content
  end

  return element
end

function Str(element)
  element.text = remove_markdown_suffix(element.text)
  return element
end

function Code(element)
  element.text = remove_markdown_suffix(element.text)
  return element
end

function Header(element)
  if not first_h1_removed and element.level == 1 then
    first_h1_removed = true
    return {}
  end
  return element
end
