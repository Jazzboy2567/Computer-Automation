/*
 * dom_summary.js — the single injected JS pass behind `get_dom_summary`.
 *
 * Runs once via page.evaluate(). It walks the live document (plus same-origin
 * iframes and OPEN shadow roots), keeps only *meaningful* nodes (interactive +
 * text-bearing), computes accessibility-tree-style data (role + accessible
 * name) the way the browser's a11y tree does, filters by visibility, flags
 * in-viewport vs needs-scroll, builds a robust locator fallback chain per
 * element, tags each kept element with `data-pilot-ref` for a fast re-find,
 * and returns a flat, document-ordered list.
 *
 * Document-order traversal => deterministic refs: the same page yields the same
 * e1, e2, ... across calls within a run, so recipes can rely on them.
 *
 * It is intentionally self-contained (no imports) and defensive (every cross-
 * frame / cross-origin access is wrapped in try/catch) so one bad node can
 * never break the whole snapshot.
 */
(opts) => {
  opts = opts || {};
  const MAX_NAME = opts.maxName || 120;
  const REF_ATTR = "data-pilot-ref";

  const notes = [];
  const nodes = [];
  let counter = 0;

  // ---- small helpers -----------------------------------------------------
  const collapse = (s) => (s || "").replace(/\s+/g, " ").trim();
  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

  const PRICE_RE = /(?:[$£€¥]\s?\d|\d+[.,]\d{2}\s?(?:USD|EUR|GBP|kr|zł)?)/i;

  const INTERACTIVE_TAGS = new Set([
    "a", "button", "input", "select", "textarea", "summary", "option", "label",
  ]);
  const INTERACTIVE_ROLES = new Set([
    "button", "link", "tab", "menuitem", "menuitemcheckbox", "menuitemradio",
    "checkbox", "radio", "switch", "option", "combobox", "textbox", "searchbox",
    "slider", "spinbutton",
  ]);
  // Tags whose *own* (direct) text we treat as readable content.
  const TEXT_TAGS = new Set([
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "dt", "dd",
    "label", "caption", "figcaption", "blockquote", "summary", "legend",
  ]);

  // Implicit ARIA role for the common tags (small, pragmatic subset).
  function implicitRole(el) {
    const tag = el.tagName.toLowerCase();
    switch (tag) {
      case "a": return el.hasAttribute("href") ? "link" : "generic";
      case "button": return "button";
      case "select": return "combobox";
      case "textarea": return "textbox";
      case "summary": return "button";
      case "option": return "option";
      case "h1": case "h2": case "h3": case "h4": case "h5": case "h6":
        return "heading";
      case "li": return "listitem";
      case "td": return "cell";
      case "th": return "columnheader";
      case "img": return "img";
      case "input": {
        const t = (el.getAttribute("type") || "text").toLowerCase();
        if (["button", "submit", "reset", "image"].includes(t)) return "button";
        if (t === "checkbox") return "checkbox";
        if (t === "radio") return "radio";
        if (t === "range") return "slider";
        if (["search"].includes(t)) return "searchbox";
        if (["text", "email", "tel", "url", "password", "number"].includes(t))
          return "textbox";
        return "textbox";
      }
      default: return null;
    }
  }

  function roleOf(el) {
    const explicit = el.getAttribute && el.getAttribute("role");
    if (explicit) return explicit.split(/\s+/)[0];
    return implicitRole(el) || "generic";
  }

  // Direct (own) text: text that lives directly in this element, not in
  // descendants. Lets us capture headings/cells/prices without duplicating the
  // text of interactive descendants (which are captured on their own).
  function directText(el) {
    let s = "";
    for (const n of el.childNodes) {
      if (n.nodeType === 3) s += n.nodeValue; // text node
    }
    return collapse(s);
  }

  // Accessible name, following the spirit of the ARIA name computation.
  function accName(el) {
    try {
      const labelledby = el.getAttribute && el.getAttribute("aria-labelledby");
      if (labelledby) {
        const txt = labelledby
          .split(/\s+/)
          .map((id) => {
            const t = el.getRootNode().getElementById
              ? el.getRootNode().getElementById(id)
              : document.getElementById(id);
            return t ? collapse(t.innerText || t.textContent) : "";
          })
          .filter(Boolean)
          .join(" ");
        if (txt) return txt;
      }
      const label = el.getAttribute && el.getAttribute("aria-label");
      if (label) return collapse(label);

      const tag = el.tagName.toLowerCase();
      if (tag === "input") {
        const type = (el.getAttribute("type") || "text").toLowerCase();
        if (["button", "submit", "reset"].includes(type) && el.value)
          return collapse(el.value);
      }
      // Associated <label> for form controls.
      if (["input", "select", "textarea"].includes(tag)) {
        if (el.id) {
          const lab = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
          if (lab) return collapse(lab.innerText || lab.textContent);
        }
        const wrap = el.closest && el.closest("label");
        if (wrap) return collapse(wrap.innerText || wrap.textContent);
        const ph = el.getAttribute("placeholder");
        if (ph) return collapse(ph);
      }
      if (tag === "img") {
        const alt = el.getAttribute("alt");
        if (alt) return collapse(alt);
      }
      // Links/buttons/headings/options: visible text content.
      const txt = collapse(el.innerText || el.textContent);
      if (txt) return txt;
      const title = el.getAttribute && el.getAttribute("title");
      if (title) return collapse(title);
    } catch (e) {
      /* defensive: never let one node break the pass */
    }
    return "";
  }

  function isAriaHidden(el) {
    let n = el;
    while (n && n.nodeType === 1) {
      if (n.getAttribute && n.getAttribute("aria-hidden") === "true") return true;
      if (n.hasAttribute && n.hasAttribute("hidden")) return true;
      n = n.parentElement || (n.getRootNode() && n.getRootNode().host);
    }
    return false;
  }

  function isVisible(el) {
    try {
      if (isAriaHidden(el)) return false;
      // Chromium's checkVisibility covers display/visibility/content-visibility.
      if (el.checkVisibility) {
        if (!el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }))
          return false;
      } else {
        const st = getComputedStyle(el);
        if (st.display === "none" || st.visibility === "hidden") return false;
      }
      const rect = el.getClientRects();
      if (!rect || rect.length === 0) return false;
      const r = el.getBoundingClientRect();
      if (r.width < 1 && r.height < 1) return false;
      return true;
    } catch (e) {
      return false;
    }
  }

  function inViewport(el) {
    const r = el.getBoundingClientRect();
    const vh = window.innerHeight || document.documentElement.clientHeight;
    const vw = window.innerWidth || document.documentElement.clientWidth;
    return r.bottom > 0 && r.right > 0 && r.top < vh && r.left < vw;
  }

  function keyAttrs(el) {
    const out = {};
    const tag = el.tagName.toLowerCase();
    const href = el.getAttribute && el.getAttribute("href");
    if (href) out.href = clip(href, 120);
    if (["input", "textarea", "select"].includes(tag)) {
      const type = el.getAttribute("type");
      if (type) out.type = type;
      const ph = el.getAttribute("placeholder");
      if (ph) out.placeholder = clip(ph, 60);
      if (typeof el.value === "string" && el.value && tag !== "password")
        out.value = clip(el.value, 60);
      if (el.checked != null && (type === "checkbox" || type === "radio"))
        out.checked = String(el.checked);
    }
    if (el.getAttribute && el.getAttribute("aria-expanded"))
      out.expanded = el.getAttribute("aria-expanded");
    return out;
  }

  // ---- robust locator chain (priority order) -----------------------------
  function cssPath(el) {
    // Build a bounded nth-of-type path, anchoring on the nearest id.
    const parts = [];
    let n = el;
    let depth = 0;
    while (n && n.nodeType === 1 && depth < 8) {
      if (n.id) {
        parts.unshift(`#${CSS.escape(n.id)}`);
        break;
      }
      const tag = n.tagName.toLowerCase();
      const parent = n.parentElement;
      if (!parent) {
        parts.unshift(tag);
        break;
      }
      const sibs = Array.from(parent.children).filter(
        (c) => c.tagName === n.tagName
      );
      if (sibs.length > 1) {
        parts.unshift(`${tag}:nth-of-type(${sibs.indexOf(n) + 1})`);
      } else {
        parts.unshift(tag);
      }
      n = parent;
      depth++;
    }
    return parts.join(" > ");
  }

  function uniqueText(el, role, name) {
    if (!name || name.length > 60) return null;
    try {
      // Heuristic uniqueness: count elements whose trimmed text equals name.
      const all = Array.from(el.ownerDocument.querySelectorAll("a,button,[role]"));
      const matches = all.filter(
        (c) => collapse(c.innerText || c.textContent) === name
      );
      if (matches.length === 1) return name;
    } catch (e) {}
    return null;
  }

  function buildLocators(el, role, name) {
    const locs = [];
    const ds =
      el.getAttribute("data-testid") ||
      el.getAttribute("data-test") ||
      el.getAttribute("data-cy");
    if (ds) locs.push(`css=[data-testid="${ds}"]`);
    if (el.id) {
      try {
        if (el.ownerDocument.querySelectorAll(`#${CSS.escape(el.id)}`).length === 1)
          locs.push(`css=#${CSS.escape(el.id)}`);
      } catch (e) {}
    }
    if (role && role !== "generic" && name) {
      locs.push(`role=${role}|name=${clip(name, 80)}`);
    }
    const ut = uniqueText(el, role, name);
    if (ut) locs.push(`text=${ut}`);
    locs.push(`css=${cssPath(el)}`);
    return locs;
  }

  // ---- decide whether to keep a node -------------------------------------
  function isInteractive(el) {
    const tag = el.tagName.toLowerCase();
    if (INTERACTIVE_TAGS.has(tag)) {
      if (tag === "a" && !el.hasAttribute("href")) return false;
      if (tag === "input") {
        const t = (el.getAttribute("type") || "text").toLowerCase();
        if (t === "hidden") return false;
      }
      if (tag === "label") return false; // labels captured as text, not clicks
      if (tag === "option") return false; // options are part of their select
      return true;
    }
    const role = el.getAttribute("role");
    if (role && INTERACTIVE_ROLES.has(role.split(/\s+/)[0])) return true;
    if (el.hasAttribute("onclick")) return true;
    const ce = el.getAttribute("contenteditable");
    if (ce === "" || ce === "true") return true;
    const ti = el.getAttribute("tabindex");
    if (ti != null && ti !== "-1" && el.children.length === 0) return true;
    return false;
  }

  function textKind(el) {
    const tag = el.tagName.toLowerCase();
    const dt = directText(el);
    if (!dt) return null;
    if (TEXT_TAGS.has(tag)) return dt;
    if (PRICE_RE.test(dt) && dt.length <= 40) return dt; // capture prices anywhere
    return null;
  }

  // ---- the traversal -----------------------------------------------------
  function record(el, kind, name, framePath) {
    counter += 1;
    const ref = "e" + counter;
    try {
      el.setAttribute(REF_ATTR, ref);
    } catch (e) {}
    const role = roleOf(el);
    const vis = inViewport(el);
    const node = {
      ref,
      role,
      name: clip(name, MAX_NAME),
      tag: el.tagName.toLowerCase(),
      kind,
      attrs: keyAttrs(el),
      in_viewport: vis,
      locators: buildLocators(el, role, name),
      frame_path: framePath,
    };
    nodes.push(node);
  }

  function walk(root, framePath) {
    let el = root.firstElementChild
      ? root
      : root; // root may be a Document or Element
    // Use a TreeWalker over elements for document order.
    const doc = root.ownerDocument || root;
    const start = root.documentElement || root;
    const tw = doc.createTreeWalker(start, NodeFilter.SHOW_ELEMENT, null);
    let cur = start;
    while (cur) {
      try {
        if (cur.nodeType === 1 && cur !== start && isVisible(cur)) {
          const tag = cur.tagName.toLowerCase();
          if (tag === "iframe" || tag === "frame") {
            // same-origin iframe traversal; cross-origin throws -> note it.
            let childDoc = null;
            try {
              childDoc = cur.contentDocument;
            } catch (e) {
              childDoc = null;
            }
            if (childDoc && childDoc.documentElement) {
              walk(childDoc, framePath.concat([nodes.length]));
            } else {
              notes.push("cross-origin/inaccessible iframe skipped (use vision)");
            }
          } else if (isInteractive(cur)) {
            record(cur, "interactive", accName(cur), framePath);
          } else {
            const t = textKind(cur);
            if (t) record(cur, "text", t, framePath);
          }
          // Open shadow root: traverse it inline.
          if (cur.shadowRoot) {
            try {
              walk(cur.shadowRoot, framePath);
            } catch (e) {
              notes.push("shadow root traversal error");
            }
          }
        }
      } catch (e) {
        /* skip a bad node, keep going */
      }
      cur = tw.nextNode();
    }
  }

  // shadowRoot has no documentElement; handle root that is a ShadowRoot.
  function walkRoot(root, framePath) {
    if (root.nodeType === 11 /* DocumentFragment / ShadowRoot */) {
      const tw = (root.ownerDocument || document).createTreeWalker(
        root,
        NodeFilter.SHOW_ELEMENT,
        null
      );
      let cur = tw.nextNode();
      while (cur) {
        try {
          if (isVisible(cur)) {
            if (isInteractive(cur)) record(cur, "interactive", accName(cur), framePath);
            else {
              const t = textKind(cur);
              if (t) record(cur, "text", t, framePath);
            }
            if (cur.shadowRoot) walkRoot(cur.shadowRoot, framePath);
          }
        } catch (e) {}
        cur = tw.nextNode();
      }
    } else {
      walk(root, framePath);
    }
  }

  // Patch: route shadow roots through walkRoot.
  const origWalk = walk;
  walk = function (root, framePath) {
    if (root.nodeType === 11) return walkRoot(root, framePath);
    return origWalk(root, framePath);
  };

  try {
    walk(document, []);
  } catch (e) {
    notes.push("top-level traversal error: " + (e && e.message));
  }

  const interactiveCount = nodes.filter((n) => n.kind === "interactive").length;
  return {
    url: location.href,
    title: document.title,
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight,
      scroll_x: window.scrollX,
      scroll_y: window.scrollY,
      page_height: Math.max(
        document.body ? document.body.scrollHeight : 0,
        document.documentElement ? document.documentElement.scrollHeight : 0
      ),
    },
    nodes,
    notes: Array.from(new Set(notes)),
    counts: { total: nodes.length, interactive: interactiveCount },
  };
}
