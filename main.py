import flet as ft
from filters import *
from images import *

import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from tkinter import simpledialog

def main(page: ft.Page):
    page.title = "Trabalho PDI"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.START
    page.window_width = 560
    page.window_height = 690

    def read_dialog(message):
        return simpledialog.askfloat("Input", message)

    def get_current_img():
        return cv2.imread('original.jpg')

    def get_new_img():
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        cv2.imwrite('./original.jpg', cv2.imread(filename))

    def set_img():
        cv2.imwrite('./original.jpg', cv2.imread('./filtered.jpg'))

    def write_filtered(img):
        cv2.imwrite('./filtered.jpg', img)
        cv2.imshow('Imagem Filtrada', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    page.add(
        ft.Column(
        [
            ft.Row(
                [
                    ft.Container(
                        content=ft.Text("Filtros e técnicas", size=40),
                        margin=1,
                        padding=10,
                        alignment=ft.alignment.center_left,
                        width=360,
                        height=100,
                        border_radius=10,
                        ink=True,
                    ),
                    ft.Container(
                        content=ft.Icon(name=ft.icons.FILE_UPLOAD),
                        margin=1,
                        padding=10,
                        alignment=ft.alignment.center,
                        bgcolor=ft.colors.PURPLE_400,
                        width=70,
                        height=60,
                        border_radius=10,
                        ink=True,
                        on_click=lambda e: get_new_img(),
                    ),
                    ft.Container(
                        content=ft.Icon(name=ft.icons.FILE_DOWNLOAD),
                        margin=1,
                        padding=10,
                        alignment=ft.alignment.center,
                        bgcolor=ft.colors.PURPLE_400,
                        width=70,
                        height=60,
                        border_radius=10,
                        ink=True,
                        on_click=lambda e: set_img(),
                    ),
                ],
            ),
            ft.Row(
                [
                ft.Container(
                    content=ft.Text("Ajuste de Gamma"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(adjust_gamma(get_current_img(), read_dialog("Intensidade: "))),
                    ),
                ft.Container(
                    content=ft.Text("Passa-Alta (Basic)"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_high_pass_filter_basic(get_current_img())),
                    ),
                ft.Container(
                    content=ft.Text("Passa-Baixa (Basic)"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_low_pass_filter_mean(get_current_img(), read_dialog("Kernel: "))),
                    ),
                ]
            ),
            ft.Row(
                [
                ft.Container(
                    content=ft.Text("Limiarização"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_threshold_filter(get_current_img(), read_dialog("Threshold: "))),
                    ),
                ft.Container(
                    content=ft.Text("Passa-Alta (Alto-Reforço)"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_high_pass_filter_boost(get_current_img(), read_dialog("Fator de Reforço: "))),
                    ),
                ft.Container(
                    content=ft.Text("Passa-Baixa (Mediana)"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_low_pass_filter_median(get_current_img(), read_dialog("Kernel: "))),
                    ),
                ]
            ),
            ft.Row(
                [
                ft.Container(
                    content=ft.Text("Escala de Cinza"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_greyscale_filter(get_current_img())),
                    ),
                ft.Container(
                    content=ft.Text("Roberts"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_roberts_filter(get_current_img())),
                    ),
                ft.Container(
                    content=ft.Text("Prewitt"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_prewitt_filter(get_current_img())),
                    ),
                ]
            ),
            ft.Row(
                [
                ft.Container(
                    content=ft.Text("Sobel"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_sobel_filter(get_current_img())),
                    ),
                ft.Container(
                    content=ft.Text("Log"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_log_filter(get_current_img())),
                    ),
                ft.Container(
                    content=ft.Text("Zerocross"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(find_zero_crossings(apply_log_filter(get_current_img()))),
                    ),
                ]
            ),
            ft.Row(
                [
                ft.Container(
                    content=ft.Text("Canny"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_canny_filter(get_current_img(), read_dialog("Limiar Inferior"), read_dialog("Limiar Superior"))),
                    ),
                ft.Container(
                    content=ft.Text("Salt & Pepper"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(add_salt_and_pepper_noise(get_current_img(), read_dialog("Intensidade: "))),
                    ),
                ft.Container(
                    content=ft.Text("Watershed"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_watershed_filter(get_current_img())),
                    ),
                ]
            ),
            ft.Row(
                [
                ft.Container(
                    content=ft.Text("Histograma (Escala de cinza)"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: plot_histogram(get_current_img()),
                    ),
                ft.Container(
                    content=ft.Text("Ajuste adaptativo de histograma"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(equalize_histogram(get_current_img())),
                    ),
                ft.Container(
                    content=ft.Text("Contagem de Objetos"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.PURPLE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(count_objects(get_current_img())),
                    ),
                ],
            ),
        ],
        )
    )

ft.app(target=main)