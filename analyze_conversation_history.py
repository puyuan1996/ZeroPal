import sqlite3


def analyze_conversation_history():
    """
    分析对话历史数据库中的数据
    """
    # 连接到SQLite数据库
    conn = sqlite3.connect('database/conversation_history.db')
    c = conn.cursor()

    # 获取总的对话记录数
    c.execute("SELECT COUNT(*) FROM history")
    total_records = c.fetchone()[0]
    print(f"总对话记录数: {total_records}")

    # 获取不同用户的对话记录数
    c.execute("SELECT user_id, COUNT(*) as count FROM history GROUP BY user_id")
    user_records = c.fetchall()
    print("每个用户的对话记录数:")
    for user_id, count in user_records:
        print(f"用户 {user_id}: {count} 条记录")

    # 获取平均对话轮数
    c.execute("SELECT AVG(cnt) FROM (SELECT user_id, COUNT(*) as cnt FROM history GROUP BY user_id)")
    avg_turns = c.fetchone()[0]
    print(f"平均对话轮数: {avg_turns}")

    # 获取最长的用户输入和助手输出
    c.execute("SELECT MAX(LENGTH(user_input)) FROM history")
    max_user_input_length = c.fetchone()[0]
    print(f"最长的用户输入: {max_user_input_length} 个字符")

    c.execute("SELECT MAX(LENGTH(assistant_output)) FROM history")
    max_assistant_output_length = c.fetchone()[0]
    print(f"最长的助手输出: {max_assistant_output_length} 个字符")

    # 关闭游标
    c.close()
    # 关闭数据库连接
    conn.close()


def clear_context():
    """
    清除对话历史
    """
    # 连接到SQLite数据库
    conn = sqlite3.connect('conversation_history.db')
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    return "", "", ""


def get_history():
    """
    获取对话历史记录
    """
    # 连接到SQLite数据库
    conn = sqlite3.connect('conversation_history.db')
    c = conn.cursor()
    c.execute("SELECT user_input, assistant_output FROM history")
    rows = c.fetchall()
    history = ""
    for row in rows:
        history += f"User: {row[0]}\nAssistant: {row[1]}\n\n"
    return history