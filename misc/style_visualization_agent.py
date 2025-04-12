import os
from openai import OpenAI


CODE  = """import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return x**2
x = np.linspace(-2, 2, 400)
y = f(x)
x1, x2 = -1.5, 1.5
y1, y2 = f(x1), f(x2)
t = np.linspace(0, 1, 100)
x_line = x1 * (1 - t) + x2 * t
y_line = y1 * (1 - t) + y2 * t
plt.plot(x, y, label="Convex function $f(x) = x^2$")
plt.plot(x_line, y_line, 'r--', label="Chord between two points")
plt.scatter([x1, x2], [y1, y2], color='red')
plt.title("Demonstration of Convexity")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
"""

def visual_refinement_prompt(code):
    system_prompt = (
        "You are a visualization artist specializing in educational graphics and data presentation. "
        "Your expertise lies in transforming technically correct visualizations into visually compelling, "
        "aesthetically pleasing educational tools through expert application of color theory, typography, "
        "spacing, and visual hierarchy principles."
        # f"Your goal is to make sure that the code aligns with the {goal}"
    )
    
    user_prompt = (
        f"Enhance the visual appeal and instructional effectiveness of this code:\n\n{code}\n\n"
        "Apply these specific refinements while preserving all functionality:\n"
        "1. Improve COLOR SCHEME using a harmonious palette that enhances conceptual understanding\n"
        "2. Refine TYPOGRAPHY with appropriate font sizes, weights, and styles for hierarchical clarity\n"
        "3. Optimize LAYOUT with proper spacing, alignment, and proportions\n"
        "4. Enhance ANNOTATIONS with strategic positioning and visual connections to relevant elements\n"
        "5. Add POLISH with appropriate figure size, aspect ratio, and resolution\n\n"
        "Return ONLY the refined code without explanation. Your enhancements should be subtle yet impactful, "
        "focusing on making the visualization more engaging and easier to comprehend."
    )
    
    # return [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": user_prompt}
    # ]
    return system_prompt, user_prompt

exec(CODE)
instructions, prompt = visual_refinement_prompt(CODE)

response = client.responses.create(
    model="gpt-4o-mini",
    instructions=instructions,
    input=prompt,
    )

print(response)
refined_code = response.output[0].content[0].text
if refined_code.startswith("```"):
    refined_code = refined_code.strip("```python").strip("```").strip()
    exec(refined_code)



