window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, latlng) {
                return L.circleMarker(latlng, {
                    radius: 6,
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.7
                });
            }

            ,
        function1: function(feature, layer) {
            if (!feature || !feature.properties) {
                return;
            }
            const p = feature.properties;

            // Cluster feature (supercluster).
            if (!!p.cluster) {
                const n = p.point_count || 0;
                const html = `<b>Cluster</b><br>Cases: ${n.toLocaleString()}<br><span style="color:#6c757d">Zoom in to see individual cases.</span>`;
                layer.bindTooltip(html, {
                    sticky: true,
                    direction: "top"
                });
                layer.bindPopup(html);
                return;
            }

            // Individual case point.
            const dept = p.departamento ? `Departamento: ${p.departamento}<br>` : "";
            const mun = p.municipio ? `Municipio: ${p.municipio}<br>` : "";
            const sex = p.sex ? `Sex: ${p.sex}<br>` : "";
            const ag = p.age_group ? `Age group: ${p.age_group}<br>` : "";
            const age = (p.age_years !== undefined && p.age_years !== null && p.age_years !== "") ? `Age: ${p.age_years}<br>` : "";
            const date = p.fecha_de_muerte ? `Death date: ${p.fecha_de_muerte}<br>` : "";

            const html = `<b>Case location</b><br>` + dept + mun + sex + ag + age + date;

            layer.bindTooltip(html, {
                sticky: true,
                direction: "top"
            });
            layer.bindPopup(html);
        }

    }
});