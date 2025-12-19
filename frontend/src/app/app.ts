import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';

interface Message {
  role: 'user' | 'assistant';
  text: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule
  ],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  API_URL = 'http://localhost:8000';

  query = '';
  indexing = false;

  messages = signal<Message[]>([]);

  constructor(private http: HttpClient) {}

  sendQuery() {
    if (!this.query.trim()) return;

    const question = this.query;
    this.query = '';

    this.messages.update(m => [
      ...m,
      { role: 'user', text: question }
    ]);

    this.http.post<{ answer: string }>(
      `${this.API_URL}/query`,
      { query: question }
    ).subscribe(res => {
      this.messages.update(m => [
        ...m,
        { role: 'assistant', text: res.answer }
      ]);
    });
  }

  indexDocuments() {
    this.indexing = true;

    this.http.post<{ indexed_count: number }>(
      `${this.API_URL}/index-documents`,
      {}
    ).subscribe(res => {
      this.messages.update(m => [
        ...m,
        {
          role: 'assistant',
          text: `📚 Проиндексировано документов: ${res.indexed_count}`
        }
      ]);
      this.indexing = false;
    }, () => this.indexing = false);
  }
}
