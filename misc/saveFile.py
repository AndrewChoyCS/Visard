def save_visualization(self, code):
    try:
        cleaned_code = code.strip().replace('```python', '').replace('```', '').strip()
        
        local_namespace = {}
        exec(cleaned_code, globals(), local_namespace)
        plt.tight_layout()
        figure_path = os.path.join('research_results', 'gradient_descent_visualization.png')
        if plt.gcf().get_axes():
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close() 
            self.logger.info(f"Visualization saved to {figure_path}")
            return figure_path
        else:
            self.logger.warning("No figure was generated to save.")
            return None
    
    except Exception as e:
        self.logger.error(f"Error saving visualization: {e}")
        traceback.print_exc()
        return None

def create_pdf(self, image_path, blurb):
    try:
        pdf_path = os.path.join('research_results', f'{self.TOPIC}_visualization_report.pdf')
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, 
                                rightMargin=72, leftMargin=72, 
                                topMargin=72, bottomMargin=18)
        
        story = []
        
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        normal_style = styles['Normal']
        
        story.append(Paragraph(f"{self.TOPIC} Visualization", title_style))
        
        if image_path and os.path.exists(image_path):
            img = Image(image_path, width=6*inch, height=4*inch)
            img.hAlign = 'CENTER'
            story.append(img)
        
        story.append(Paragraph("<br/><br/>Learning Insights:", styles['Heading3']))
        story.append(Paragraph(blurb, normal_style))
        # Build PDF
        doc.build(story)
        
        self.logger.info(f"PDF report created at {pdf_path}")
    except Exception as e:
        self.logger.error(f"Error creating PDF: {e}")