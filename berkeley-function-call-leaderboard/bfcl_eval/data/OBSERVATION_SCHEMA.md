# BFCL Glaive Dataset - Observation Schema Documentation

This document provides comprehensive documentation of observation schemas for each API in the BFCL Glaive dataset.

**Dataset File**: `bfcl_glaive_dataset.json`  
**Total Records**: 800  
**Generated**: December 22, 2025

## Overview

For reinforcement learning consistency, observations should maintain a **fixed schema** with the same keys across all turns. This ensures that the RL agent always sees the same observation space structure, regardless of which specific operation is executed.

### Consistency Status

| API | Status | Keys | Notes |
|-----|--------|------|-------|
| GorillaFileSystem | ✅ **Fully Consistent** | 4 | All observations have identical keys |
| VehicleControlAPI | ✅ **Fully Consistent** | 18 | All observations have identical keys |
| TwitterAPI | ✅ **Fully Consistent** | 5 | All observations have identical keys |
| MessageAPI | ✅ **Fully Consistent** | 5 | All observations have identical keys |
| TicketAPI | ✅ **Fully Consistent** | 6 | All observations have identical keys |
| MathAPI | ✅ **Fully Consistent** | 2 | All observations have identical keys |
| TradingBot | ✅ **Fully Consistent** | 8 | All observations have identical keys |
| TravelAPI | ✅ **Fully Consistent** | 7 | All observations have identical keys |
| WebSearchAPI | ✅ **Fully Consistent** | 2 | All observations have identical keys |
| MemoryAPI_kv | ✅ **Fully Consistent** | 4 | All observations have identical keys |
| MemoryAPI_vector | ✅ **Fully Consistent** | 4 | All observations have identical keys |
| MemoryAPI_rec_sum | ✅ **Fully Consistent** | 4 | All observations have identical keys |

## Table of Contents

1. [GorillaFileSystem](#gorillafilesystem) ✅
2. [VehicleControlAPI](#vehiclecontrolapi) ✅
3. [TwitterAPI](#twitterapi) ✅
4. [MessageAPI](#messageapi) ✅
5. [TicketAPI](#ticketapi) ✅
6. [MathAPI](#mathapi) ✅
7. [TradingBot](#tradingbot) ✅
8. [TravelAPI](#travelapi) ✅
9. [WebSearchAPI](#websearchapi) ✅
10. [MemoryAPI](#memoryapi) ✅

---

## GorillaFileSystem

### Description

The GorillaFileSystem API simulates a Linux-style filesystem interface with operations for file and directory management. It maintains a virtual filesystem state including current working directory and directory contents. This API is commonly used in multi-turn scenarios requiring file organization, search, and manipulation tasks.

**Use Cases**: File organization, directory navigation, text file operations, search and filtering

### Observation Schema

✅ **Consistency Status: FULLY CONSISTENT**

All observations for GorillaFileSystem contain exactly **4 keys** across all turns and all operations:

#### State Keys (Filesystem State)
- **`current_working_directory`** (string): Current directory path (e.g., "/workspace/document")
- **`directory_contents`** (array): List of files and directories in the current directory

#### Result Keys (Operation Feedback)
- **`result`** (string): Operation-specific result message
- **`context`** (string): Human-readable context and description of what happened

### Available Tools

**Total Tools**: 18

- `cd` - Change directory
- `ls` - List directory contents
- `cat` - Display file contents
- `grep` - Search for patterns in files
- `find` - Find files matching pattern
- `mkdir` - Create directory
- `touch` - Create empty file
- `rm` - Remove file
- `rmdir` - Remove directory
- `mv` - Move/rename file
- `cp` - Copy file
- `wc` - Count lines/words/characters
- `tail` - Display last lines of file
- `sort` - Sort file contents
- `diff` - Compare two files
- `du` - Disk usage
- `echo` - Print text
- `pwd` - Print working directory

### Real Examples from Dataset

#### Example 1: `cd` - Directory Navigation

**Function Call**:
```json
{
  "name": "cd",
  "arguments": {
    "folder": "document"
  }
}
```

**Observation**:
```json
{
  "current_working_directory": "/workspace/document",
  "directory_contents": [
    "final_report.pdf",
    "previous_report.pdf"
  ],
  "result": "Changed to directory: document",
  "context": "Changed to directory: document"
}
```

**Analysis**: All 4 keys present. State is updated to reflect new directory and its contents.

*(Source: Record #0, Turn 0)*

#### Example 2: `mkdir` - Directory Creation

**Function Call**:
```json
{
  "name": "mkdir",
  "arguments": {
    "dir_name": "temp"
  }
}
```

**Observation**:
```json
{
  "current_working_directory": "/workspace/document",
  "directory_contents": [
    "final_report.pdf",
    "previous_report.pdf",
    "temp"
  ],
  "result": "Directory 'temp' created successfully",
  "context": "Directory 'temp' created successfully"
}
```

**Analysis**: All 4 keys present. `directory_contents` updated to show the new directory.

*(Source: Record #0, Turn 0)*

#### Example 3: `grep` - Pattern Search

**Function Call**:
```json
{
  "name": "grep",
  "arguments": {
    "file_name": "final_report.pdf",
    "pattern": "budget analysis"
  }
}
```

**Observation**:
```json
{
  "current_working_directory": "/workspace/document/temp",
  "directory_contents": [
    "final_report.pdf"
  ],
  "result": "Found 1 matching lines",
  "context": "Matching lines: ['Year2024 This is the final report content including budget analysis and other sections.']"
}
```

**Analysis**: All 4 keys present. Operation details are in `context`, maintaining schema consistency.

*(Source: Record #0, Turn 1)*

---

## VehicleControlAPI

### Description

The VehicleControlAPI simulates a comprehensive vehicle control system. It manages vehicle state including fuel, battery, engine, doors, climate control, lights, brakes, and navigation. This API is used in scenarios involving vehicle operations, trip planning, and system monitoring.

**Use Cases**: Vehicle control, trip planning, fuel management, safety system monitoring, climate control

### Observation Schema

✅ **Consistency Status: FULLY CONSISTENT**

All observations for VehicleControlAPI contain exactly **18 keys** across all turns and all operations:

#### Core Vehicle State
- **`fuel_level`** (float): Current fuel level in gallons
- **`battery_voltage`** (float): Battery voltage in volts
- **`engine_state`** (string): Engine status ("running" or "stopped")
- **`door_status`** (object): Status of all doors with keys: driver, passenger, rear_left, rear_right

#### Climate Control State
- **`ac_temperature`** (float): AC temperature setting in degrees Celsius
- **`fan_speed`** (integer): Fan speed setting (0-100)
- **`ac_mode`** (string): AC mode ("auto", "cool", "heat", "defrost")

#### Lights and Brakes
- **`headlight_status`** (string): Headlight status ("on" or "off")
- **`parking_brake_status`** (string): Parking brake status ("engaged" or "released")
- **`brake_pedal_status`** (string): Brake pedal status ("pressed" or "released")

#### Tire Pressure
- **`front_left_tire_pressure`** (float): Front left tire pressure in PSI
- **`front_right_tire_pressure`** (float): Front right tire pressure in PSI
- **`rear_left_tire_pressure`** (float): Rear left tire pressure in PSI
- **`rear_right_tire_pressure`** (float): Rear right tire pressure in PSI

#### Navigation and Cruise Control
- **`cruise_status`** (string): Cruise control status ("active" or "inactive")
- **`destination`** (string): Current navigation destination

#### Result Keys
- **`result`** (string): Operation-specific result message
- **`context`** (string): Human-readable context/description

### Available Tools

**Total Tools**: 22

- `activateParkingBrake` - Engage/release parking brake
- `adjustClimateControl` - Adjust AC settings
- `check_tire_pressure` - Check tire pressures
- `displayCarStatus` - Display specific system status
- `display_log` - Display vehicle logs
- `estimate_distance` - Calculate distance between locations
- `estimate_drive_feasibility_by_mileage` - Check if trip is feasible
- `fillFuelTank` - Add fuel to tank
- `find_nearest_tire_shop` - Locate tire shops
- `gallon_to_liter` - Convert gallons to liters
- `get_current_speed` - Get vehicle speed
- `get_outside_temperature_from_google` - Get temperature from Google
- `get_outside_temperature_from_weather_com` - Get temperature from Weather.com
- `get_zipcode_based_on_city` - Get zipcode for city
- `liter_to_gallon` - Convert liters to gallons
- `lockDoors` - Lock/unlock doors
- `pressBrakePedal` - Press brake pedal
- `releaseBrakePedal` - Release brake pedal
- `setCruiseControl` - Set cruise control
- `setHeadlights` - Turn headlights on/off
- `set_navigation` - Set navigation destination
- `startEngine` - Start/stop engine

### Real Examples from Dataset

#### Example 1: `lockDoors` - Door Control

**Function Call**:
```json
{
  "name": "lockDoors",
  "arguments": {
    "unlock": true,
    "door": [
      "driver",
      "passenger",
      "rear_left",
      "rear_right"
    ]
  }
}
```

**Observation**:
```json
{
  "fuel_level": 10.5,
  "battery_voltage": 12.6,
  "engine_state": "stopped",
  "door_status": {
    "driver": "unlocked",
    "passenger": "unlocked",
    "rear_left": "unlocked",
    "rear_right": "unlocked"
  },
  "ac_temperature": 22.0,
  "fan_speed": 70,
  "ac_mode": "auto",
  "headlight_status": "off",
  "parking_brake_status": "released",
  "brake_pedal_status": "released",
  "front_left_tire_pressure": 32.0,
  "front_right_tire_pressure": 32.0,
  "rear_left_tire_pressure": 30.0,
  "rear_right_tire_pressure": 30.0,
  "cruise_status": "inactive",
  "destination": "None",
  "result": "Operation lockDoors completed",
  "context": "lockStatus: unlocked | remainingUnlockedDoors: 4"
}
```

**Analysis**: All 18 keys present. Notice how `door_status` is updated to reflect the operation, while all other state keys remain present with their current values.

*(Source: Record #50, Turn 0)*

#### Example 2: `setHeadlights` - Light Control

**Function Call**:
```json
{
  "name": "setHeadlights",
  "arguments": {
    "mode": "on"
  }
}
```

**Observation**:
```json
{
  "fuel_level": 10.5,
  "battery_voltage": 12.6,
  "engine_state": "stopped",
  "door_status": {
    "driver": "unlocked",
    "passenger": "unlocked",
    "rear_left": "unlocked",
    "rear_right": "unlocked"
  },
  "ac_temperature": 22.0,
  "fan_speed": 70,
  "ac_mode": "auto",
  "headlight_status": "on",
  "parking_brake_status": "released",
  "brake_pedal_status": "released",
  "front_left_tire_pressure": 32.0,
  "front_right_tire_pressure": 32.0,
  "rear_left_tire_pressure": 30.0,
  "rear_right_tire_pressure": 30.0,
  "cruise_status": "inactive",
  "destination": "None",
  "result": "Operation setHeadlights completed",
  "context": "headlightStatus: on"
}
```

**Analysis**: All 18 keys present. `headlight_status` changed from "off" to "on", while all other state remains consistent.

*(Source: Record #50, Turn 0)*

#### Example 3: `fillFuelTank` - Fuel Management

**Function Call**:
```json
{
  "name": "fillFuelTank",
  "arguments": {
    "fuelAmount": 15.0
  }
}
```

**Observation**:
```json
{
  "fuel_level": 22.5,
  "battery_voltage": 12.6,
  "engine_state": "stopped",
  "door_status": {
    "driver": "locked",
    "passenger": "locked",
    "rear_left": "locked",
    "rear_right": "locked"
  },
  "ac_temperature": 22.0,
  "fan_speed": 60,
  "ac_mode": "auto",
  "headlight_status": "off",
  "parking_brake_status": "released",
  "brake_pedal_status": "released",
  "front_left_tire_pressure": 35.0,
  "front_right_tire_pressure": 35.0,
  "rear_left_tire_pressure": 33.0,
  "rear_right_tire_pressure": 33.0,
  "cruise_status": "inactive",
  "destination": "None",
  "result": "Operation fillFuelTank completed",
  "context": "fuelLevel: 22.5"
}
```

**Analysis**: All 18 keys present. `fuel_level` increased from 7.5 to 22.5, demonstrating state persistence across turns.

*(Source: Record #51, Turn 1)*

---

## TwitterAPI

### Description

The TwitterAPI simulates social media posting functionality. It allows posting tweets, retweeting, commenting, and managing authentication. The API maintains user authentication state and tweet metadata.

**Use Cases**: Social media posting, content sharing, user engagement

### Observation Schema

✅ **Consistency Status: FULLY CONSISTENT**

All observations for TwitterAPI contain exactly **5 keys** across all turns and all operations:

#### State Keys (Twitter State)
- **`authenticated`** (boolean): Whether the user is currently authenticated
- **`username`** (string): Username of the authenticated user (or None if not authenticated)
- **`tweet_count`** (integer): Total number of tweets in the system

#### Result Keys (Operation Feedback)
- **`result`** (string): Operation-specific result message
- **`context`** (string): Human-readable context and description of what happened

### Available Tools

**Total Tools**: 3

- `post_tweet` - Post a new tweet with content, tags, and mentions
- `retweet` - Retweet an existing tweet
- `comment` - Comment on an existing tweet

### Real Examples from Dataset

#### Example 1: `post_tweet`

**Function Call**:
```json
{
  "name": "post_tweet",
  "arguments": {
    "content": "Initial report content More unsorted data Unsorted data",
    "mentions": [
      "@Julia"
    ],
    "tags": [
      "#currenttechtrend"
    ]
  }
}
```

**Observation**:
```json
{
  "authenticated": true,
  "username": "tech_guru",
  "tweet_count": 3,
  "result": "Operation post_tweet completed",
  "context": "Tweet posted with ID 2 | Tags: #currenttechtrend | Mentions: @Julia"
}
```

**Analysis**: All 5 keys present. State reflects authentication status, username, and total tweet count. Operation details are in context.

*(Source: Record #4)*

---

## MessageAPI

### Description

The MessageAPI simulates a messaging system for sending and receiving messages between users. It tracks message history and supports viewing sent/received messages.

**Use Cases**: Inter-user communication, message management, notification systems

### Observation Schema

✅ **Consistency Status: FULLY CONSISTENT**

All observations for MessageAPI contain exactly **5 keys** across all turns and all operations:

#### State Keys (Message System State)
- **`login_status`** (boolean): Whether a user is currently logged in
- **`message_count_sent`** (integer): Total number of messages sent by the current user
- **`message_count_received`** (integer): Total number of messages received
- **`total_contacts`** (integer): Total number of unique contacts interacted with

#### Result Keys (Operation Feedback)
- **`result`** (string): Operation-specific result message
- **`context`** (string): Human-readable context and description of what happened

### Available Tools

**Total Tools**: 5

- `send_message` - Send a message to another user
- `view_messages_sent` - View messages sent by current user
- `view_messages_received` - View messages received by current user
- `delete_message` - Delete a message
- `message_login` - Login to messaging system

### Real Examples from Dataset

#### Example 1: `send_message`

**Function Call**:
```json
{
  "name": "send_message",
  "arguments": {
    "receiver_id": "USR005",
    "message": "Kelly Total Score: 96"
  }
}
```

**Observation**:
```json
{
  "login_status": true,
  "message_count_sent": 1,
  "message_count_received": 3,
  "total_contacts": 4,
  "result": "Operation send_message completed",
  "context": "Message sent: true | Message ID: {'new_id': 67410}"
}
```

**Analysis**: All 5 keys present. State reflects login status and message counts. Operation details are in context.

*(Source: Record #17)*

---

## TicketAPI

### Description

The TicketAPI simulates a ticket management system for tracking issues and support requests. It supports creating, retrieving, editing, resolving, and closing tickets with priority levels.

**Use Cases**: Issue tracking, support ticket management, workflow automation

### Observation Schema

✅ **Consistency Status: FULLY CONSISTENT**

All observations for TicketAPI contain exactly **6 keys** across all turns and all operations:

#### State Keys (Ticket System State)
- **`login_status`** (boolean): Whether a user is currently logged in
- **`total_tickets`** (integer): Total number of tickets in the system
- **`open_tickets`** (integer): Number of tickets with "Open" status
- **`resolved_tickets`** (integer): Number of tickets with "Resolved" status
- **`closed_tickets`** (integer): Number of tickets with "Closed" status

#### Result Keys (Operation Feedback)
- **`result`** (string): Operation-specific result message
- **`context`** (string): Human-readable context and description of what happened

### Available Tools

**Total Tools**: 6

- `create_ticket` - Create a new support ticket
- `get_ticket` - Retrieve ticket details
- `resolve_ticket` - Mark ticket as resolved
- `close_ticket` - Close a ticket
- `edit_ticket` - Edit ticket details
- `ticket_login` - Login to ticket system

### Real Examples from Dataset

#### Example 1: `get_ticket`

**Function Call**:
```json
{
  "name": "get_ticket",
  "arguments": {
    "ticket_id": 987654
  }
}
```

**Observation**:
```json
{
  "login_status": true,
  "total_tickets": 5,
  "open_tickets": 3,
  "resolved_tickets": 1,
  "closed_tickets": 1,
  "result": "Operation get_ticket completed",
  "context": "Ticket 987654 retrieved | Status: open"
}
```

**Analysis**: All 6 keys present. State reflects login status and ticket counts by status. Operation details are in context.

*(Source: Record #24)*

---

## MathAPI

### Description

The MathAPI provides mathematical computation functions including statistical operations (mean, standard deviation) and logarithms with configurable precision. This is a stateless API that performs pure calculations without maintaining state.

**Use Cases**: Statistical calculations, mathematical operations, data analysis

### Observation Schema

✅ **Consistency Status: FULLY CONSISTENT**

All observations for MathAPI contain exactly **2 keys** across all turns and all operations:

#### Result Keys (Operation Feedback)
- **`result`** (number): The calculated result value
- **`context`** (string): Human-readable description of the calculation performed

**Note**: MathAPI is stateless but maintains consistent schema with result and context keys.

### Available Tools

**Total Tools**: 3

- `mean` - Calculate arithmetic mean of numbers
- `standard_deviation` - Calculate standard deviation
- `logarithm` - Calculate logarithm with specified base and precision

### Real Examples from Dataset

#### Example 1: `mean`

**Function Call**:
```json
{
  "name": "mean",
  "arguments": {
    "numbers": [
      3,
      16,
      60
    ]
  }
}
```

**Observation**:
```json
{
  "result": 26.333333333333332,
  "context": "Mean calculated: 26.333333333333332"
}
```

**Analysis**: All 2 keys present. Consistent schema with result and context.

*(Source: Record #15)*

#### Example 2: `logarithm`

**Function Call**:
```json
{
  "name": "logarithm",
  "arguments": {
    "value": 980.0,
    "base": 20,
    "precision": 10
  }
}
```

**Observation**:
```json
{
  "result": 2.3027397666,
  "context": "Logarithm calculated: 2.3027397666"
}
```

**Analysis**: Same consistent schema across different math operations.

*(Source: Record #59)*

---

## TradingBot

### Description

The TradingBot API simulates a stock trading system for executing trades, managing accounts, and tracking portfolio information. It maintains account state, order history, watchlists, and transaction records.

**Use Cases**: Stock trading, portfolio management, account management, market analysis

### Observation Schema

✅ **Consistency Status: FULLY CONSISTENT**

All observations for TradingBot contain exactly **8 keys** across all turns and all operations:

#### State Keys (Trading System State)
- **`authenticated`** (boolean): Whether the user is currently authenticated
- **`account_balance`** (float): Current account balance in USD
- **`account_id`** (integer): Account identifier
- **`market_status`** (string): Current market status ("Open" or "Closed")
- **`total_orders`** (integer): Total number of orders in the system
- **`watch_list_count`** (integer): Number of stocks in the watchlist
- **`transaction_count`** (integer): Total number of transactions in history

#### Result Keys (Operation Feedback)
- **`result`** (string): Operation-specific result message
- **`context`** (string): Human-readable context and description of what happened

### Available Tools

**Total Tools**: 22

- `get_current_time` - Get current time
- `get_symbol_by_name` - Get stock symbol by company name
- `get_stock_info` - Get stock details (price, volume, moving averages)
- `get_order_details` - Get order information
- `cancel_order` - Cancel an order
- `place_order` - Place a buy/sell order
- `withdraw_funds` - Withdraw funds from account
- `get_account_info` - Get account information
- `trading_login` - Login to trading system
- `trading_get_login_status` - Check login status
- `trading_logout` - Logout from trading system
- `fund_account` - Add funds to account
- `remove_stock_from_watchlist` - Remove stock from watchlist
- `get_watchlist` - Get watchlist
- `get_order_history` - Get order history
- `get_transaction_history` - Get transaction history
- `get_available_stocks` - Get stocks by sector
- `filter_stocks_by_price` - Filter stocks by price range
- `add_to_watchlist` - Add stock to watchlist
- `notify_price_change` - Check for significant price changes

---

## TravelAPI

### Description

The TravelAPI simulates a travel booking system for managing flight bookings, credit cards, and travel budgets. It maintains booking records, credit card information, and user authentication state.

**Use Cases**: Flight booking, travel management, budget tracking, credit card management

### Observation Schema

✅ **Consistency Status: FULLY CONSISTENT**

All observations for TravelAPI contain exactly **7 keys** across all turns and all operations:

#### State Keys (Travel System State)
- **`authenticated`** (boolean): Whether the user is currently authenticated (token valid)
- **`total_bookings`** (integer): Total number of flight bookings
- **`total_credit_cards`** (integer): Number of registered credit cards
- **`budget_limit`** (float or null): Budget limit in USD (if set)
- **`user_first_name`** (string or null): User's first name
- **`user_last_name`** (string or null): User's last name

#### Result Keys (Operation Feedback)
- **`result`** (string): Operation-specific result message
- **`context`** (string): Human-readable context and description of what happened

### Available Tools

**Total Tools**: 16

- `authenticate_travel` - Authenticate user with travel API
- `travel_get_login_status` - Check authentication status
- `get_budget_fiscal_year` - Get budget fiscal year
- `register_credit_card` - Register a credit card
- `get_flight_cost` - Get flight costs for route
- `get_credit_card_balance` - Get credit card balance
- `book_flight` - Book a flight
- `retrieve_invoice` - Retrieve booking invoice
- `get_booking_history` - Get all booking history
- `list_all_airports` - List available airports
- `cancel_booking` - Cancel a booking
- `compute_exchange_rate` - Convert currencies
- `verify_traveler_information` - Verify traveler details
- `set_budget_limit` - Set budget limit
- `get_nearest_airport_by_city` - Find nearest airport
- `purchase_insurance` - Purchase travel insurance
- `contact_customer_support` - Contact support
- `get_all_credit_cards` - Get all registered credit cards

---

## WebSearchAPI

### Description

The WebSearchAPI provides web search functionality using search engines and URL content fetching. This is a stateless API that performs search queries and retrieves web content without maintaining persistent state.

**Use Cases**: Web search, information retrieval, content fetching

### Observation Schema

✅ **Consistency Status: FULLY CONSISTENT**

All observations for WebSearchAPI contain exactly **2 keys** across all turns and all operations:

#### Result Keys (Operation Feedback)
- **`result`** (string or list): Search results or content retrieved
- **`context`** (string): Human-readable description of the operation

**Note**: WebSearchAPI is stateless but maintains consistent schema with result and context keys.

### Available Tools

**Total Tools**: 2

- `search_engine_query` - Search the web with keywords
- `fetch_url_content` - Fetch content from a URL (raw, markdown, or truncated)

---

## MemoryAPI

### Description

The MemoryAPI provides memory management functionality with three variants: key-value storage, vector embeddings, and recursive summarization. Each variant maintains different types of memory state for agent interactions.

**Use Cases**: Agent memory management, information storage and retrieval, context management

### Observation Schema

✅ **Consistency Status: FULLY CONSISTENT**

All MemoryAPI observations maintain consistent schemas, but the exact keys depend on the variant:

#### MemoryAPI_kv (Key-Value Variant)

All observations contain exactly **4 keys**:

- **`core_memory_count`** (integer): Number of entries in short-term (core) memory
- **`archival_memory_count`** (integer): Number of entries in long-term (archival) memory
- **`result`** (string): Operation-specific result message
- **`context`** (string): Human-readable context and description

#### MemoryAPI_vector (Vector Embedding Variant)

All observations contain exactly **4 keys**:

- **`core_memory_count`** (integer): Number of entries in short-term (core) memory
- **`archival_memory_count`** (integer): Number of entries in long-term (archival) memory
- **`result`** (string): Operation-specific result message
- **`context`** (string): Human-readable context and description

#### MemoryAPI_rec_sum (Recursive Summary Variant)

All observations contain exactly **4 keys**:

- **`memory_length`** (integer): Length of the memory string in characters
- **`memory_empty`** (boolean): Whether the memory is empty
- **`result`** (string): Operation-specific result message
- **`context`** (string): Human-readable context and description

### Available Tools

**Total Tools**: Varies by variant

#### MemoryAPI_kv and MemoryAPI_vector
- `core_memory_add` / `core_memory_remove` / `core_memory_replace` / `core_memory_clear`
- `core_memory_retrieve` / `core_memory_list_keys` / `core_memory_key_search` / `core_memory_retrieve_all`
- `archival_memory_add` / `archival_memory_remove` / `archival_memory_replace` / `archival_memory_clear`
- `archival_memory_retrieve` / `archival_memory_list_keys` / `archival_memory_key_search`
- `core_memory_update` / `archival_memory_update` (vector variant only)

#### MemoryAPI_rec_sum
- `memory_append` - Append text to memory
- `memory_update` - Replace memory content
- `memory_clear` - Clear memory
- `memory_replace` - Replace specific text in memory
- `memory_retrieve` - Retrieve current memory content

---

## Summary

### All APIs Are Now Fully Consistent ✅

**All 12 APIs in the BFCL benchmark now have fully consistent observation schemas**, making them ideal for reinforcement learning:

1. **GorillaFileSystem** (4 keys) ✅
   - Every operation returns exactly the same 4 keys
   - State is always observable (current directory + contents)
   - Result and context provide operation feedback

2. **VehicleControlAPI** (18 keys) ✅
   - Every operation returns exactly the same 18 keys
   - Complete vehicle state is always observable
   - Rich state representation for complex control tasks

3. **TwitterAPI** (5 keys) ✅
   - Every operation returns exactly the same 5 keys
   - Authentication status, username, and tweet count always present

4. **MessageAPI** (5 keys) ✅
   - Every operation returns exactly the same 5 keys
   - Login status and message counts always observable

5. **TicketAPI** (6 keys) ✅
   - Every operation returns exactly the same 6 keys
   - Login status and ticket counts by status always present

6. **MathAPI** (2 keys) ✅
   - Every operation returns exactly the same 2 keys
   - Consistent schema for stateless calculations

7. **TradingBot** (8 keys) ✅
   - Every operation returns exactly the same 8 keys
   - Account state, market status, and order counts always observable

8. **TravelAPI** (7 keys) ✅
   - Every operation returns exactly the same 7 keys
   - Authentication, bookings, and credit card counts always present

9. **WebSearchAPI** (2 keys) ✅
   - Every operation returns exactly the same 2 keys
   - Consistent schema for stateless search operations

10. **MemoryAPI_kv** (4 keys) ✅
    - Every operation returns exactly the same 4 keys
    - Core and archival memory counts always observable

11. **MemoryAPI_vector** (4 keys) ✅
    - Every operation returns exactly the same 4 keys
    - Core and archival memory counts always observable

12. **MemoryAPI_rec_sum** (4 keys) ✅
    - Every operation returns exactly the same 4 keys
    - Memory length and empty status always present

---

## Key Consistency Benefits

The consistent observation schemas across all APIs provide several advantages for RL training:

1. **Fixed Observation Space**: The agent always sees the same state representation for each API
2. **State Tracking**: Core state variables are always present, even if null
3. **Operation Agnostic**: Different operations return the same schema
4. **Training Stability**: Consistent inputs lead to more stable learning
5. **Policy Generalization**: The model can learn state-action mappings more effectively
6. **Cross-API Consistency**: All APIs follow the same pattern, making multi-API training easier

### Recommendations

For optimal RL training with consistent observation spaces:
- ✅ **All APIs are ready for RL training** - Every API has perfect consistency
- ✅ **Use any API confidently** - All maintain fixed observation schemas
- 📊 **Monitor schema consistency** - Verify observation keys match expectations during training
- 🔄 **Leverage cross-API patterns** - Consistent schemas enable better transfer learning

---

**End of Documentation**
