import trimesh
import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['PYGAME_DISPLAY'] = '0'

# Загрузка 3D модели человека
mesh = trimesh.load_mesh('human.obj')

# Печать информации о модели
print("Модель загружена:")
print("Вершины модели:", mesh.vertices)
print("Грани модели:", mesh.faces)

# Инициализация Pygame для рендеринга
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
clock = pygame.time.Clock()

# Настройки OpenGL
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -20)  # Отдаляем камеру

# Включение освещения
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLight(GL_LIGHT0, GL_POSITION,  (1, 1, 1, 0))  # Позиция источника света
glLight(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1))  # Атмосферное освещение
glLight(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))  # Основное освещение

# Функция для загрузки данных модели в VBO
def load_vbo(mesh):
    vertices = mesh.vertices
    faces = mesh.faces

    # Создание и загрузка данных в VBO
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    face_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, face_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)

    return vertex_buffer, face_buffer

# Функция для рендеринга с использованием VBO
def render_vbo(vertex_buffer, face_buffer):
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glEnableClientState(GL_VERTEX_ARRAY)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, face_buffer)
    glDrawElements(GL_TRIANGLES, 3 * len(mesh.faces), GL_UNSIGNED_INT, None)

    glDisableClientState(GL_VERTEX_ARRAY)

# Загружаем данные модели в VBO
vertex_buffer, face_buffer = load_vbo(mesh)

# Масштабируем модель (если она слишком большая или маленькая)
#mesh.vertices *= 0.01  # Уменьшаем размер модели

# Основной цикл для рендеринга
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Очистка экрана и установка фона
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Чёрный фон

    # Рендеринг модели с использованием VBO
    render_vbo(vertex_buffer, face_buffer)

    # Обновление экрана
    pygame.display.flip()
    clock.tick(60)  # Частота кадров

pygame.quit()
