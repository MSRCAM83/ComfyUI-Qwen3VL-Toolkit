"""
Generate a professional fleet/rocket icon for the LoRA Fleet Pipeline launcher.
Creates a 256x256 .ico file with a stylized rocket and network nodes.
"""

from PIL import Image, ImageDraw

def create_fleet_icon():
    # Create 256x256 image with dark background
    size = 256
    img = Image.new('RGBA', (size, size), (26, 26, 46, 255))  # #1a1a2e
    draw = ImageDraw.Draw(img)

    # Color palette
    bg_dark = (26, 26, 46)
    rocket_orange = (255, 140, 0)  # #ff8c00
    rocket_gold = (255, 165, 0)    # #ffa500
    accent_blue = (70, 130, 180)

    # Draw stylized rocket ship (centered, pointing up)
    rocket_x = size // 2
    rocket_y = size // 2 - 10

    # Rocket body (elongated triangle)
    rocket_body = [
        (rocket_x, rocket_y - 50),      # tip
        (rocket_x - 20, rocket_y + 30), # bottom left
        (rocket_x + 20, rocket_y + 30)  # bottom right
    ]
    draw.polygon(rocket_body, fill=rocket_orange, outline=rocket_gold, width=2)

    # Rocket fins (two triangular fins)
    left_fin = [
        (rocket_x - 20, rocket_y + 10),
        (rocket_x - 35, rocket_y + 35),
        (rocket_x - 20, rocket_y + 30)
    ]
    right_fin = [
        (rocket_x + 20, rocket_y + 10),
        (rocket_x + 35, rocket_y + 35),
        (rocket_x + 20, rocket_y + 30)
    ]
    draw.polygon(left_fin, fill=rocket_gold, outline=rocket_orange, width=2)
    draw.polygon(right_fin, fill=rocket_gold, outline=rocket_orange, width=2)

    # Rocket window (small circle)
    window_y = rocket_y - 20
    draw.ellipse([rocket_x - 8, window_y - 8, rocket_x + 8, window_y + 8],
                 fill=accent_blue, outline=rocket_gold, width=2)

    # Exhaust flame (three flame shapes)
    flame_y = rocket_y + 30
    # Center flame
    center_flame = [
        (rocket_x, flame_y),
        (rocket_x - 8, flame_y + 20),
        (rocket_x, flame_y + 25),
        (rocket_x + 8, flame_y + 20)
    ]
    draw.polygon(center_flame, fill=rocket_gold)

    # Left flame
    left_flame = [
        (rocket_x - 12, flame_y),
        (rocket_x - 18, flame_y + 15),
        (rocket_x - 12, flame_y + 18)
    ]
    draw.polygon(left_flame, fill=rocket_orange)

    # Right flame
    right_flame = [
        (rocket_x + 12, flame_y),
        (rocket_x + 18, flame_y + 15),
        (rocket_x + 12, flame_y + 18)
    ]
    draw.polygon(right_flame, fill=rocket_orange)

    # Fleet network nodes (small circles around the rocket)
    node_positions = [
        (60, 60), (196, 60), (60, 196), (196, 196),  # corners
        (30, 128), (226, 128), (128, 30),            # edges
    ]

    for x, y in node_positions:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6],
                     fill=accent_blue, outline=rocket_gold, width=1)

    # Connection lines (subtle lines from nodes toward center)
    line_color = (70, 130, 180, 100)  # semi-transparent blue
    for x, y in node_positions:
        draw.line([(x, y), (rocket_x, rocket_y)], fill=line_color, width=1)

    # Save as .ico file
    output_path = r'C:\Users\matth\ComfyUI-Qwen3VL-Toolkit\fleet.ico'
    img.save(output_path, format='ICO', sizes=[(256, 256)])
    print(f"Icon created successfully: {output_path}")
    print("  - 256x256 pixels")
    print("  - Stylized rocket ship with fleet network")
    print("  - Orange/gold rocket on dark navy background")

if __name__ == '__main__':
    create_fleet_icon()
