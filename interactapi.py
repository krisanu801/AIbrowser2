from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import google.generativeai as genai
import os
import json
import logging
import base64
from typing import Dict, Any, Optional, List
import tempfile
import easyocr
import cv2



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize API
app = FastAPI(title="Browser Automation Agent - Interact API")

# Initialize Gemini (make sure to set GEMINI_API_KEY in environment variables)
genai.configure(api_key="AIzaSyCoGZgxiG3fxp7DLbQiDJVeZWOQGTOiNRg")
model = genai.GenerativeModel('gemini-2.0-flash')
chat = model.start_chat()

# Define screenshot directory
SCREENSHOT_DIR = os.path.join(tempfile.gettempdir(), "browser_automation_screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Global browser controller instance
browser_controller = None

class Command(BaseModel):
    """Model for natural language commands sent to the API"""
    command: str
    wait_time: Optional[int] = 5000  # Default wait time in ms
    analyze_on_failure: Optional[bool] = True  # Whether to analyze page on failure

class CommandResponse(BaseModel):
    """Model for API responses"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class BrowserController:
    """Handles browser automation using Playwright"""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.context = None
        self.status = "inactive"
        self.last_screenshot_path = None
        self.element_position = None
        self.texts = None
    
    async def start_browser(self):
        """Initialize browser instance"""
        if self.status == "active":
            return True
        
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=False)
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()
            self.status = "active"
            logger.info("Browser started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start browser: {str(e)}")
            self.status = "error"
            raise HTTPException(status_code=500, detail=f"Browser initialization failed: {str(e)}")
    
    async def close_browser(self):
        """Close browser and clean up resources"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            self.page = None
            self.context = None
            self.browser = None
            self.playwright = None
            self.status = "inactive"
            logger.info("Browser closed successfully")
            return True
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")
            return False
    
    async def take_screenshot(self, reason="debug"):
        """Take a screenshot of the current page"""
        try:
            filename = f"screenshot_{reason}_{int(time.time())}.png"
            filepath = os.path.join(SCREENSHOT_DIR, filename)
            await self.page.screenshot(path=filepath)
            self.last_screenshot_path = filepath
            logger.info(f"Screenshot saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to take screenshot: {str(e)}")
            return None
    
    async def get_screenshot_as_base64(self):
        """Take a screenshot and return as base64 encoded string"""
        try:
            screenshot_bytes = await self.page.screenshot()
            base64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
            return True, base64_image
        except Exception as e:
            error_msg = f"Failed to take screenshot: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    async def analyze_page_for_element(self, element_description):
        """
        Analyze the page using Gemini to find the proper selector for an element
        based on its description and a screenshot
        """
        try:
            # Take a screenshot
            screenshot_path = await self.take_screenshot("element_analysis")
            if not screenshot_path:
                return False, "Failed to take screenshot for analysis"
            

            reader = easyocr.Reader(['en']) # You can add more languages
            image = cv2.imread(screenshot_path)
            results = reader.readtext(image)
            textz = "Texts found in the image:\n"

            for (bbox, text, prob) in results:
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                br = (int(br[0]), int(br[1]))
                text2 = repr(text)+ f",Coordinates: {tl},"+ f"Probability: {prob} \n"
                textz += text2
            
            # Use Gemini multimodal capabilities to analyze the page
            print(textz)
            prompt = f"""
            I'm trying to find a proper selector for: "{element_description}" on this webpage whose screenshot i provided.
            
            Using the screenshot please provide:
            1. The most precise CSS selector for this element
            2. Alternative selectors (id, class, xpath) if applicable
            3. The visible text content of the element if any
            then Using only this text- {textz}  please provide:
            1. Coordinate position(x , y) of the primary selector element where we want to click 
            **get coordinates strictly from the text provided which contains the coordinates of the found texts in the image**
            Focus on finding unique identifiers that won't change with page updates.
            When giving coordinates , do not indulge in calculations simply give the one needed .
            
            Response format:
            {{
              "primary_selector": "precise_selector_here",
              "alternative_selectors": ["selector1", "selector2"],
              "element_text": "text_content_if_any",
              "element_type": "button/input/link/etc" , 
              "element_position": {{
                "x": x,
                "y": y
              }}
            }}
            
            x , y must be integers
            """

            image_file = genai.upload_file(screenshot_path)
            
            # Create a multimodal prompt
            parts = [prompt , image_file ]
            
            response = chat.send_message(parts)
            response_text = response.text
            logger.info(f"Response from Gemini: {response_text}")
            
            # Extract JSON from response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            else:
                # Try to find any JSON-like structure
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            
            try:
                element_info = json.loads(response_text)
                self.element_position = (element_info["element_position"]["x"] , element_info["element_position"]["y"])
                logger.info(f"Element analysis result: {element_info}")
                return True, element_info
            except json.JSONDecodeError:
                return False, f"Failed to parse element analysis result: {response_text}"
        
        except Exception as e:
            error_msg = f"Failed to analyze page for element: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    async def navigate(self, url):
        """Navigate to a URL"""
        try:
            await self.page.goto(url, wait_until="networkidle")
            logger.info(f"Navigated to {url}")
            return True, f"Successfully navigated to {url}"
        except PlaywrightTimeoutError:
            logger.warning(f"Navigation to {url} timed out, but continuing")
            return True, f"Navigation to {url} timed out, but continuing"
        except Exception as e:
            error_msg = f"Navigation to {url} failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    async def click(self, selector, analyze_on_failure=True):
        """Click on an element"""
        try:
            await self.page.mouse.click(self.element_position[0] + 7, self.element_position[1] + 7)
            logger.info(f"Clicked on element: {selector}")
            return True, f"Successfully clicked on element: {selector}"
        except Exception as e:
            error_msg = f"Failed to click on element {selector}: {str(e)}"
            screenshot_path = await self.take_screenshot("click_failure")
            
            if analyze_on_failure:
                logger.info(f"Attempting to analyze page to find element: {selector}")
                return False, {
                    "error": error_msg,
                    "screenshot": screenshot_path,
                    "analyze_needed": True,
                    "element_description": selector
                }
            
            return False, error_msg
    
    async def type_text(self, selector, text, analyze_on_failure=True):
        """Type text into an input field"""
        input_fields = await self.page.locator("input").evaluate_all("elements => elements.map(e => e.outerHTML)")

        # Print each field’s HTML
        textz = "fields :\n"
        for i, field in enumerate(input_fields, 1):
            textz += f"Field {i}: {field}\n"
        print(textz)
        prompt = f"""
        from this {textz} along with the value that's going to get filled {text} in selector{selector}
        identify the field that requires to get filled
        sample output :
        {{idoffield1 :texttobefilledinfield1, idoffield2:texttobefilledinfield2 , ....}}
        return only the json object , nothing more
        """

        response = chat.send_message(prompt).text  
        logger.info(f"Response from Gemini: {response}")
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        else:
            # Try to find any JSON-like structure
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
        field_dict = json.loads(response_text)
        logger.info(f"Field dictionary: {field_dict}")

        try:
            if field_dict:
                for field, name in field_dict.items():
                    await self.page.wait_for_selector(f'#{field}')
                    await self.page.fill(f'#{field}', name)
            else :
                logger.info(99999999)
                await self.page.keyboard.type(text)
            return True, f"Successfully typed text into element: {selector}"
        except Exception as e:
            error_msg = f"Failed to type text into element {selector}: {str(e)} , , , now trying different way"
            logger.error(error_msg)
            await self.page.keyboard.type(text)
            
            # Take screenshot on failure
            screenshot_path = await self.take_screenshot("type_failure")
            
            if analyze_on_failure:
                logger.info(f"Attempting to analyze page to find element: {selector}")
                return False, {
                    "error": error_msg,
                    "screenshot": screenshot_path,
                    "analyze_needed": True,
                    "element_description": selector
                }
            
            return True, f"Successfully typed text into element: {selector}"
    
    async def wait_for_selector(self, selector, timeout=5000):
        """Wait for an element to appear"""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            logger.info(f"Element found: {selector}")
            return True, f"Element found: {selector}"
        except Exception as e:
            error_msg = f"Timeout waiting for element {selector}: {str(e)}"
            logger.error(error_msg)
            
            # Take screenshot on failure
            screenshot_path = await self.take_screenshot("wait_failure")
            
            return False, error_msg
    
    async def press_key(self, key):
        """Press a keyboard key"""
        try:
            await self.page.keyboard.press(key)
            logger.info(f"Pressed key: {key}")
            return True, f"Successfully pressed key: {key}"
        except Exception as e:
            error_msg = f"Failed to press key {key}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    async def scroll(self, selector=None, direction="down"):
        """Scroll the page or a specific element"""
        try:
            if selector:
                element = await self.page.query_selector(selector)
                if element:
                    if direction == "down":
                        element.evaluate("el => el.scrollBy(0, 300)")
                    elif direction == "up":
                        element.evaluate("el => el.scrollBy(0, -300)")
                    logger.info(f"Scrolled element {selector} {direction}")
                    return True, f"Scrolled element {direction}"
                return False, f"Element {selector} not found for scrolling"
            else:
                if direction == "down":
                    await self.page.evaluate("window.scrollBy(0, 300)")
                elif direction == "up":
                    await self.page.evaluate("window.scrollBy(0, -300)")
                logger.info(f"Scrolled page {direction}")
                return True, f"Scrolled page {direction}"
        except Exception as e:
            error_msg = f"Failed to scroll {direction}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    async def get_page_title(self):
        """Get the current page title"""
        try:
            title = await self.page.title()
            logger.info(f"Page title: {title}")
            return True, title
        except Exception as e:
            error_msg = f"Failed to get page title: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    async def get_current_url(self):
        """Get the current URL"""
        try:
            url = self.page.url
            logger.info(f"Current URL: {url}")
            return True, url
        except Exception as e:
            error_msg = f"Failed to get current URL: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    async def get_element_info(self, selector):
        """Get detailed information about an element"""
        try:
            element = await self.page.query_selector(selector)
            if not element:
                return False, f"Element {selector} not found"
            
            info = {}
            
            # Get tag name
            tag_name = await element.evaluate("el => el.tagName")
            info["tag_name"] = tag_name.lower()
            
            # Get text content
            text_content = await element.text_content()
            info["text_content"] = text_content.strip() if text_content else ""
            
            # Get attributes
            attrs = await element.evaluate("""el => {
                const attributes = {};
                for (const attr of el.attributes) {
                    attributes[attr.name] = attr.value;
                }
                return attributes;
            }""")
            info["attributes"] = attrs
            
            # Get dimensions and position
            bbox = await element.bounding_box()
            info["position"] = bbox
            
            # Is visible
            is_visible = await element.is_visible()
            info["is_visible"] = is_visible
            
            return True, info
        except Exception as e:
            error_msg = f"Failed to get element info for {selector}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

async def parse_command(command_text):
    """
    Use Gemini to parse a natural language command into structured actions
    """
    prompt = f"""
    Parse the following browser automation command into structured actions.
    Return a JSON object with 'action' and necessary parameters.
    
    Supported actions:
    - navigate: requires 'url' parameter
    - click: requires 'selector' parameter
    - type: requires 'selector' and 'text' parameters
    - wait: requires 'selector' parameter
    - press: requires 'key' parameter
    - scroll: optional 'selector', optional 'direction' (up/down, default: down)
    - title: get the page title (no parameters)
    - url: get the current url (no parameters)
    - screenshot: take a screenshot (no parameters)

    when providing urls in parameter values use the correct url
    
    Command: {command_text}
    
    Response format:
    {{
      "action": "action_name",
      "parameters": {{
        "param1": "value1",
        ...
      }}
    }}
    """
    
    try:
        response = chat.send_message(prompt)
        response_text = response.text
        
        # Extract JSON from response (handle potential text wrapping)
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        else:
            # Try to find any JSON-like structure
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
        
        parsed_action = json.loads(response_text)
        logger.info(f"Parsed command: {parsed_action}")
        return parsed_action
    except Exception as e:
        logger.error(f"Failed to parse command: {str(e)}")
        return {"action": "error", "parameters": {"message": f"Failed to parse command: {str(e)}"}}

async def execute_command(command_text, wait_time=5000, analyze_on_failure=True):
    """
    Execute a natural language command on the browser
    """
    global browser_controller
    
    # Initialize browser if needed
    if browser_controller is None or browser_controller.status != "active":
        browser_controller = BrowserController()
        await browser_controller.start_browser()
    
    # Parse the command
    parsed_command = await parse_command(command_text)
    
    if parsed_command["action"] == "error":
        return False, parsed_command["parameters"]["message"]
    
    action = parsed_command["action"]
    parameters = parsed_command.get("parameters", {})
    
    # Handle each action with proper awaits
    print(f"Executing action: {action} with parameters: {parameters}")
    
    # For actions that may need element analysis if they fail
    if action.lower() in ["click", "type"]:
        # First, take a screenshot to capture current state
        #screenshot_success, screenshot_result = await browser_controller.get_screenshot_as_base64()
        
        # Try the action with the given selector first
        if action.lower() == "click":
            # Use element description from the selector or a more descriptive value if available
            element_description = parameters.get("element_description", parameters.get("selector"))
            
            # Analyze the page to find the element
            analysis_success, element_info = await browser_controller.analyze_page_for_element(element_description)
            if analysis_success and isinstance(element_info, dict) and "primary_selector" in element_info:
                # Get the new selector from the analysis
                new_selector = element_info["primary_selector"]
                logger.info(f"Element analysis provided new selector: {new_selector}")
                retry_success, retry_result = await browser_controller.click(new_selector, False)
                return CommandResponse(
                    success=retry_success,
                    message=f"Element analysis provided selector: {new_selector}. Retry result: {retry_result if isinstance(retry_result, str) else 'Action executed'}",
                    data={
                        "command": command_text,
                        "original_selector": parameters.get("selector"),
                        "new_selector": new_selector,
                        "element_analysis": element_info,
                        "retry_success": retry_success
                    }
                )
            else:
                return CommandResponse(
                    success=False,
                    message=f"Failed to find element. Analysis result: {element_info if analysis_success else 'Analysis failed'}",
                    data={
                        "command": command_text,
                        "original_selector": parameters.get("selector"),
                        "element_analysis_success": analysis_success,
                        "element_analysis": element_info if analysis_success else None
                    }
                )


        elif action.lower() == "type":
            success, result = await browser_controller.type_text(parameters["selector"], parameters["text"], False)
        
        
                
                
            
        
        return success, result
    
    # For other actions that don't need element analysis
    if action.lower() == "navigate":
        return await browser_controller.navigate(parameters.get("url"))
    elif action.lower() == "wait":
        return await browser_controller.wait_for_selector(parameters.get("selector"), wait_time)
    elif action.lower() == "press":
        return await browser_controller.press_key(parameters.get("key"))
    elif action.lower() == "scroll":
        return await browser_controller.scroll(parameters.get("selector"), parameters.get("direction", "down"))
    elif action.lower() == "title":
        return await browser_controller.get_page_title()
    elif action.lower() == "url":
        return await browser_controller.get_current_url()
    elif action.lower() == "screenshot":
        return await browser_controller.get_screenshot_as_base64()
    else:
        return False, f"Unsupported action: {action}"


@app.post("/interact", response_model=CommandResponse)
async def interact_endpoint(command: Command, background_tasks: BackgroundTasks):
    """
    Endpoint to handle natural language browser automation commands
    """
    try:
        result = await execute_command(command.command, command.wait_time, command.analyze_on_failure)
        
        # Check if the result is already a CommandResponse (from element analysis flow)
        if isinstance(result, CommandResponse):
            return result
        
        # Otherwise, unpack the success and message from the result tuple
        success, message = result
        
        # Normal response
        return CommandResponse(
            success=success,
            message=message if isinstance(message, str) else "Command executed",
            data={"command": command.command, "result": message if not isinstance(message, str) else None}
        )
    except Exception as e:
        logger.error(f"Error in interact endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-browser", response_model=CommandResponse)
async def start_browser():
    """
    Endpoint to explicitly start the browser
    """
    global browser_controller
    
    try:
        if browser_controller is None:
            browser_controller = BrowserController()
        
        success = await browser_controller.start_browser()
        
        return CommandResponse(
            success=success,
            message="Browser started successfully" if success else "Failed to start browser"
        )
    except Exception as e:
        logger.error(f"Error starting browser: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/close-browser", response_model=CommandResponse)
async def close_browser():
    """
    Endpoint to explicitly close the browser
    """
    global browser_controller
    
    try:
        if browser_controller is not None:
            success = await browser_controller.close_browser()
            
            return CommandResponse(
                success=success,
                message="Browser closed successfully" if success else "Failed to close browser"
            )
        else:
            return CommandResponse(
                success=True,
                message="No active browser to close"
            )
    except Exception as e:
        logger.error(f"Error closing browser: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=CommandResponse)
async def get_status():
    """
    Endpoint to check the browser status
    """
    global browser_controller
    
    try:
        if browser_controller is not None:
            return CommandResponse(
                success=True,
                message=f"Browser status: {browser_controller.status}",
                data={"status": browser_controller.status}
            )
        else:
            return CommandResponse(
                success=True,
                message="Browser not initialized",
                data={"status": "inactive"}
            )
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/screenshot", response_model=CommandResponse)
async def take_screenshot():
    """
    Endpoint to take a screenshot of the current page
    """
    global browser_controller
    
    try:
        if browser_controller is None or browser_controller.status != "active":
            return CommandResponse(
                success=False,
                message="Browser not active"
            )
        
        success, result = await browser_controller.get_screenshot_as_base64()
        
        return CommandResponse(
            success=success,
            message="Screenshot taken successfully" if success else "Failed to take screenshot",
            data={"screenshot_base64": result} if success else {"error": result}
        )
    except Exception as e:
        logger.error(f"Error taking screenshot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-element", response_model=CommandResponse)
async def analyze_element(description: str):
    """
    Endpoint to analyze the page and find an element based on description
    """
    global browser_controller
    
    try:
        if browser_controller is None or browser_controller.status != "active":
            return CommandResponse(
                success=False,
                message="Browser not active"
            )
        
        success, result = await browser_controller.analyze_page_for_element(description)
        
        return CommandResponse(
            success=success,
            message="Element analysis successful" if success else "Failed to analyze element",
            data={"element_info": result} if success else {"error": result}
        )
    except Exception as e:
        logger.error(f"Error analyzing element: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Make sure to import time for timestamp in screenshot names
import time

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)