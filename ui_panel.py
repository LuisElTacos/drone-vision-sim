"""Panneau latéral (Dear ImGui via imgui-bundle) pour choisir les vues affichées.

On utilise seulement le moteur de rendu OpenGL 3 d'ImGui ; les entrées souris sont
transmises à la main depuis les callbacks pyGLFW (évite de mélanger deux GLFW).
"""

from imgui_bundle import imgui

PANEL_WIDTH = 270  # en pixels « fenêtre » (avant mise à l'échelle écran)


class SidePanel:
    def __init__(self):
        imgui.create_context()
        io = imgui.get_io()
        io.set_ini_filename("")  # pas de fichier imgui.ini
        imgui.style_colors_dark()
        imgui.backends.opengl3_init("#version 330")
        self.mouse_enabled = True

    # --- entrées --------------------------------------------------------------

    def mouse_pos(self, x, y):
        if self.mouse_enabled:
            imgui.get_io().add_mouse_pos_event(x, y)

    def mouse_button(self, button, pressed):
        if button < 3:
            imgui.get_io().add_mouse_button_event(button, pressed)

    def set_mouse_enabled(self, on):
        """Désactive la souris pour ImGui pendant le pilotage (curseur capturé)."""
        self.mouse_enabled = on
        if not on:
            imgui.get_io().add_mouse_pos_event(-1e9, -1e9)

    def wants_mouse(self):
        return imgui.get_io().want_capture_mouse

    # --- frame ----------------------------------------------------------------

    def draw(self, window_size, framebuffer_size, dt, views, info_lines, settings=None):
        """views : liste de [libellé, actif] (modifiée sur place)."""
        io = imgui.get_io()
        ww, wh = window_size
        fw, fh = framebuffer_size
        io.display_size = imgui.ImVec2(ww, wh)
        io.display_framebuffer_scale = imgui.ImVec2(fw / max(ww, 1), fh / max(wh, 1))
        io.delta_time = max(dt, 1e-4)

        imgui.backends.opengl3_new_frame()
        imgui.new_frame()
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(imgui.ImVec2(PANEL_WIDTH, wh))
        flags = (imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize
                 | imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_title_bar)
        imgui.begin("Vues", None, flags)
        imgui.text("Vues affichées")
        imgui.separator()
        for view in views:
            _, view[1] = imgui.checkbox(view[0], view[1])

        if settings is not None:
            imgui.spacing()
            imgui.separator()
            imgui.text("Réglages de vol")

            # Vitesse max
            speed_val = settings.get("max_speed", 8.0)
            imgui.text(f"Vitesse max : {speed_val * 3.6:.0f} km/h")
            imgui.set_next_item_width(-1)
            _, settings["max_speed"] = imgui.slider_float(
                "##max_speed", float(speed_val), 2.0, 30.0, "%.1f m/s"
            )

            # Sensibilité lacet (rotation)
            yaw_val = settings.get("yaw_sens", 1.0)
            imgui.text("Sensibilité rotation :")
            imgui.set_next_item_width(-1)
            _, settings["yaw_sens"] = imgui.slider_float(
                "##yaw_sens", float(yaw_val), 0.2, 3.0, "x%.2f"
            )

            # Contrôleur de vitesse : réactivité aux changements de direction et freinage
            for key, label, lo, hi, fmt in (
                    ("ctrl_kv", "Réactivité (gain vitesse) :", 1.0, 15.0, "%.1f /s"),
                    ("ctrl_ki", "Gain intégral :", 0.0, 3.0, "%.2f"),
                    ("ctrl_tilt", "Inclinaison max / freinage :", 10.0, 60.0, "%.0f °")):
                imgui.text(label)
                imgui.set_next_item_width(-1)
                _, settings[key] = imgui.slider_float("##" + key, float(settings[key]), lo, hi, fmt)

            # Sensibilité caméra
            cam_val = settings.get("cam_sens", 1.0)
            imgui.text("Sensibilité caméra :")
            imgui.set_next_item_width(-1)
            _, settings["cam_sens"] = imgui.slider_float(
                "##cam_sens", float(cam_val), 0.2, 3.0, "x%.2f"
            )

            imgui.spacing()
            imgui.separator()
            imgui.text("Caméras")
            _, settings["cams_on_chassis"] = imgui.checkbox(
                "Fixées au châssis", settings.get("cams_on_chassis", True))

            imgui.spacing()
            imgui.separator()
            imgui.text("Contrôles")
            _, settings["gamepad_enabled"] = imgui.checkbox("Piloter à la manette", settings.get("gamepad_enabled", False))
            if settings.get("gamepad_connected", False):
                name = settings.get("gamepad_name", "Manette")
                if len(name) > 22:
                    name = name[:20] + "..."
                imgui.text_colored(imgui.ImVec4(0.3, 0.9, 0.4, 1.0), f"[OK] {name}")
            else:
                imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "(Aucune manette)")

            imgui.spacing()
            imgui.separator()
            imgui.text("Environnement")
            _, settings["show_grid"] = imgui.checkbox("Quadriller le sol", settings.get("show_grid", True))

            imgui.spacing()
            imgui.text("Environnement :")
            mode = settings.get("building_mode", 0)
            if imgui.radio_button("Murs troués", mode == 1):
                settings["building_mode"] = 1
            if imgui.radio_button("Forêt", mode == 3):
                settings["building_mode"] = 3
            if imgui.radio_button("Bâtiment (vol intérieur)", mode == 0):
                settings["building_mode"] = 0
            if imgui.radio_button("Vide", mode == 2):
                settings["building_mode"] = 2

            if mode == 1:
                imgui.spacing()
                imgui.separator()
                imgui.text("Murs troués")
                btn_w = PANEL_WIDTH - 25
                if imgui.button("Nouveaux murs (graine aléatoire)", imgui.ImVec2(btn_w, 24)):
                    settings["regen_maze"] = True
                if imgui.button("Retour au départ", imgui.ImVec2(btn_w, 24)):
                    settings["teleport"] = settings.get("maze_spawn")
                imgui.spacing()
                for line in settings.get("maze_info", []):
                    imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), line)
                imgui.text_colored(imgui.ImVec4(0.3, 0.9, 0.4, 1.0), "Cible : sphère verte")

            if mode == 3:
                imgui.spacing()
                imgui.separator()
                imgui.text("Forêt")
                btn_w = PANEL_WIDTH - 25
                if imgui.button("Nouvelle forêt (graine aléatoire)", imgui.ImVec2(btn_w, 24)):
                    settings["regen_forest"] = True
                _, settings["forest_density"] = imgui.slider_float(
                    "Densité (arbres/100 m²)", settings["forest_density"], 1.0, 10.0, "%.1f")
                if imgui.is_item_deactivated_after_edit():  # régénère au relâchement du slider
                    settings["regen_forest_density"] = True
                if imgui.button("Retour au départ##foret", imgui.ImVec2(btn_w, 24)):
                    settings["teleport"] = settings.get("forest_spawn")
                imgui.spacing()
                for line in settings.get("forest_info", []):
                    imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), line)
                imgui.text_colored(imgui.ImVec4(0.3, 0.9, 0.4, 1.0), "Cible : sphère verte")

            if mode == 0:
                imgui.spacing()
                imgui.separator()
                imgui.text("Téléportation drone")
                btn_w = PANEL_WIDTH - 25
                if imgui.button("Devant le portail hangar", imgui.ImVec2(btn_w, 24)):
                    settings["teleport"] = ((-2.7, 1.4, 2.0), -90.0)
                if imgui.button("Intérieur du grand hall", imgui.ImVec2(btn_w, 24)):
                    settings["teleport"] = ((-2.5, 1.5, -11.0), -90.0)
                if imgui.button("Étage mezzanine", imgui.ImVec2(btn_w, 24)):
                    settings["teleport"] = ((-3.0, 4.2, -14.0), -90.0)
                if imgui.button("Atelier adjacent", imgui.ImVec2(btn_w, 24)):
                    settings["teleport"] = ((4.5, 1.5, -11.0), -90.0)
                if imgui.button("Au-dessus du toit", imgui.ImVec2(btn_w, 24)):
                    settings["teleport"] = ((-2.5, 7.2, -10.0), -90.0)

                imgui.spacing()
                imgui.text_colored(imgui.ImVec4(0.4, 0.8, 1.0, 1.0), "25 ouvertures (vérité terrain)")
                imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), "16 franchissables, 2 niches")
                imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), "1 trappe sol, 1 puits toit")

        imgui.spacing()
        imgui.separator()
        for line in info_lines:
            imgui.text(line)
        imgui.end()
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    def shutdown(self):
        imgui.backends.opengl3_shutdown()
        imgui.destroy_context()
