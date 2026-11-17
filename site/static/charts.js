// Crosshair + tooltip for timeline charts. Each chart variant (wide / narrow) carries its own JSON data.
(function () {
  document.querySelectorAll("figure.chart[data-timeline]").forEach(function (fig) {
    var tip = fig.querySelector(".tip");
    fig.querySelectorAll(".v").forEach(function (v) {
      var data = JSON.parse(v.querySelector("script[type='application/json']").textContent);
      var svg = v.querySelector("svg"), line = svg.querySelector(".crosshair");
      var box = data.box, n = data.dates.length, cur = n - 1;
      function show(i) {
        var x = box.x0 + (box.x1 - box.x0) * (n === 1 ? 0 : i / (n - 1));
        line.setAttribute("x1", x); line.setAttribute("x2", x); line.style.display = "";
        var html = "<b>" + data.labels[i] + "</b>";
        data.series.forEach(function (s) {
          html += '<span style="--c:' + s.color + '"><em>' + s.name + "</em>" + s.fmt[i] + "</span>";
        });
        tip.innerHTML = html; tip.classList.add("on");
        var r = svg.getBoundingClientRect(), f = fig.getBoundingClientRect(), scale = r.width / data.width;
        var left = (r.left - f.left) + x * scale + 12;
        if (left + 190 > f.width) left = (r.left - f.left) + x * scale - 202;
        tip.style.left = Math.max(0, left) + "px"; tip.style.top = (r.top - f.top + box.y0 * scale) + "px";
      }
      function hide() { tip.classList.remove("on"); line.style.display = "none"; }
      function idx(evt) {
        var r = svg.getBoundingClientRect(), x = (evt.clientX - r.left) * data.width / r.width;
        return Math.max(0, Math.min(n - 1, Math.round((x - box.x0) / (box.x1 - box.x0) * (n - 1))));
      }
      svg.addEventListener("pointermove", function (e) { cur = idx(e); show(cur); });
      svg.addEventListener("pointerleave", hide);
      svg.setAttribute("tabindex", "0");
      svg.addEventListener("focus", function () { show(cur); });
      svg.addEventListener("blur", hide);
      svg.addEventListener("keydown", function (e) {
        if (e.key === "ArrowLeft") { cur = Math.max(0, cur - 1); show(cur); e.preventDefault(); }
        if (e.key === "ArrowRight") { cur = Math.min(n - 1, cur + 1); show(cur); e.preventDefault(); }
      });
    });
  });
})();
